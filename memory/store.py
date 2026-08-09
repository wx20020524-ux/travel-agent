"""
用户偏好持久化存储 —— 基于 SQLite。

表结构:
  - user_profiles: user_id, preferences (JSON), created_at, updated_at
  - conversations: id, user_id, session_id, role, content, timestamp, is_summarized
  - session_summaries: id, user_id, session_id, summary_text, structured_prefs (JSON), created_at

用法:
    store = UserProfileStore("data/memory.db")
    store.update_preferences("user_001", {"transport": ["自驾"], "hotel_type": "豪华型"})
    prefs = store.get_preferences("user_001")
    store.add_message("user_001", "sess_1", "user", "我想去杭州玩")
"""

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UserPreferences:
    """用户旅行偏好结构"""
    budget_level: str = "中等"           # 经济 / 中等 / 豪华
    transport: List[str] = field(default_factory=list)   # ["自驾", "公共交通"]
    hotel_type: str = ""                 # 经济型 / 豪华型 / 民宿
    interests: List[str] = field(default_factory=list)   # ["自然风光", "历史文化"]
    special_requirements: str = ""       # 额外要求（如带老人、无障碍）
    favorite_cities: List[str] = field(default_factory=list)  # 常去城市
    travel_style: str = ""              # 悠闲 / 紧凑 / 深度游

    def to_dict(self) -> Dict[str, Any]:
        return {
            "budget_level": self.budget_level,
            "transport": self.transport,
            "hotel_type": self.hotel_type,
            "interests": self.interests,
            "special_requirements": self.special_requirements,
            "favorite_cities": self.favorite_cities,
            "travel_style": self.travel_style,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UserPreferences":
        return cls(
            budget_level=d.get("budget_level", "中等"),
            transport=d.get("transport", []),
            hotel_type=d.get("hotel_type", ""),
            interests=d.get("interests", []),
            special_requirements=d.get("special_requirements", ""),
            favorite_cities=d.get("favorite_cities", []),
            travel_style=d.get("travel_style", ""),
        )

    def merge(self, new_prefs: "UserPreferences"):
        """增量合并新偏好（不覆盖已有非空值）"""
        if new_prefs.budget_level and new_prefs.budget_level != "中等":
            self.budget_level = new_prefs.budget_level
        if new_prefs.transport:
            # 合并去重
            merged = set(self.transport) | set(new_prefs.transport)
            self.transport = list(merged)
        if new_prefs.hotel_type:
            self.hotel_type = new_prefs.hotel_type
        if new_prefs.interests:
            merged = set(self.interests) | set(new_prefs.interests)
            self.interests = list(merged)
        if new_prefs.special_requirements:
            if self.special_requirements:
                self.special_requirements += "; " + new_prefs.special_requirements
            else:
                self.special_requirements = new_prefs.special_requirements
        if new_prefs.favorite_cities:
            merged = set(self.favorite_cities) | set(new_prefs.favorite_cities)
            self.favorite_cities = list(merged)
        if new_prefs.travel_style:
            self.travel_style = new_prefs.travel_style

    def to_prompt_fragment(self) -> str:
        """生成可注入 Prompt 的用户偏好描述"""
        parts = []
        if self.budget_level and self.budget_level != "中等":
            parts.append(f"预算偏好: {self.budget_level}")
        if self.transport:
            parts.append(f"交通偏好: {'、'.join(self.transport)}")
        if self.hotel_type:
            parts.append(f"住宿偏好: {self.hotel_type}")
        if self.interests:
            parts.append(f"兴趣偏好: {'、'.join(self.interests)}")
        if self.travel_style:
            parts.append(f"旅行风格: {self.travel_style}")
        if self.special_requirements:
            parts.append(f"特殊要求: {self.special_requirements}")
        if self.favorite_cities:
            parts.append(f"常去城市: {'、'.join(self.favorite_cities)}")
        return "\n".join(parts) if parts else ""


class UserProfileStore:
    """基于 SQLite 的用户偏好 & 对话历史存储"""

    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        preferences TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        is_summarized INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE IF NOT EXISTS session_summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        summary_text TEXT NOT NULL,
                        structured_prefs TEXT NOT NULL DEFAULT '{}',
                        message_count INTEGER NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, session_id);
                    CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations(user_id, timestamp);
                    CREATE INDEX IF NOT EXISTS idx_summary_user ON session_summaries(user_id);
                """)
                conn.commit()
            finally:
                conn.close()

    # ==================== 用户偏好 ====================

    def get_preferences(self, user_id: str) -> UserPreferences:
        """获取用户偏好，不存在返回默认值"""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT preferences FROM user_profiles WHERE user_id = ?",
                    (user_id,)
                ).fetchone()
                if row:
                    return UserPreferences.from_dict(json.loads(row["preferences"]))
                return UserPreferences()
            finally:
                conn.close()

    def update_preferences(self, user_id: str, prefs: UserPreferences):
        """保存用户偏好（增量合并）"""
        existing = self.get_preferences(user_id)
        existing.merge(prefs)
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO user_profiles (user_id, preferences, created_at, updated_at)
                       VALUES (?, ?, COALESCE((SELECT created_at FROM user_profiles WHERE user_id=?), ?), ?)""",
                    (user_id, json.dumps(existing.to_dict(), ensure_ascii=False), user_id, now, now)
                )
                conn.commit()
            finally:
                conn.close()

    def set_preferences_full(self, user_id: str, prefs: UserPreferences):
        """全量覆盖用户偏好（不合并）"""
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO user_profiles (user_id, preferences, created_at, updated_at)
                       VALUES (?, ?, COALESCE((SELECT created_at FROM user_profiles WHERE user_id=?), ?), ?)""",
                    (user_id, json.dumps(prefs.to_dict(), ensure_ascii=False), user_id, now, now)
                )
                conn.commit()
            finally:
                conn.close()

    # ==================== 对话历史 ====================

    def add_message(self, user_id: str, session_id: str, role: str, content: str):
        """添加一条对话消息"""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO conversations (user_id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (user_id, session_id, role, content, time.time())
                )
                conn.commit()
            finally:
                conn.close()

    def add_messages_batch(self, user_id: str, session_id: str,
                           messages: List[Dict[str, str]]):
        """批量添加对话消息"""
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            try:
                rows = [(user_id, session_id, m["role"], m["content"], now)
                        for m in messages]
                conn.executemany(
                    "INSERT INTO conversations (user_id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                    rows
                )
                conn.commit()
            finally:
                conn.close()

    def get_messages(self, user_id: str, session_id: Optional[str] = None,
                     limit: int = 50, before_id: Optional[int] = None) -> List[Dict]:
        """获取对话历史消息"""
        with self._lock:
            conn = self._get_conn()
            try:
                if session_id:
                    if before_id:
                        rows = conn.execute(
                            """SELECT id, role, content, timestamp FROM conversations
                               WHERE user_id=? AND session_id=? AND id < ? AND is_summarized=0
                               ORDER BY id DESC LIMIT ?""",
                            (user_id, session_id, before_id, limit)
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            """SELECT id, role, content, timestamp FROM conversations
                               WHERE user_id=? AND session_id=? AND is_summarized=0
                               ORDER BY id DESC LIMIT ?""",
                            (user_id, session_id, limit)
                        ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT id, role, content, timestamp FROM conversations
                           WHERE user_id=? AND is_summarized=0
                           ORDER BY id DESC LIMIT ?""",
                        (user_id, limit)
                    ).fetchall()
                # 反转回正序
                rows = list(reversed(rows))
                return [{"id": r["id"], "role": r["role"], "content": r["content"],
                         "timestamp": r["timestamp"]} for r in rows]
            finally:
                conn.close()

    def get_recent_sessions(self, user_id: str, limit: int = 5) -> List[str]:
        """获取最近的会话 ID 列表"""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT DISTINCT session_id FROM conversations
                       WHERE user_id=? ORDER BY session_id DESC LIMIT ?""",
                    (user_id, limit)
                ).fetchall()
                return [r["session_id"] for r in rows]
            finally:
                conn.close()

    def mark_summarized(self, message_ids: List[int]):
        """标记消息已被总结"""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executemany(
                    "UPDATE conversations SET is_summarized=1 WHERE id=?",
                    [(mid,) for mid in message_ids]
                )
                conn.commit()
            finally:
                conn.close()

    # ==================== 会话总结 ====================

    def save_summary(self, user_id: str, session_id: str, summary_text: str,
                     structured_prefs: Dict[str, Any], message_count: int):
        """保存会话总结"""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO session_summaries (user_id, session_id, summary_text,
                       structured_prefs, message_count, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (user_id, session_id, summary_text,
                     json.dumps(structured_prefs, ensure_ascii=False),
                     message_count, time.time())
                )
                conn.commit()
            finally:
                conn.close()

    def get_summaries(self, user_id: str, limit: int = 5) -> List[Dict]:
        """获取历史会话总结"""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT session_id, summary_text, structured_prefs, message_count, created_at
                       FROM session_summaries WHERE user_id=? ORDER BY created_at DESC LIMIT ?""",
                    (user_id, limit)
                ).fetchall()
                return [{
                    "session_id": r["session_id"],
                    "summary": r["summary_text"],
                    "prefs": json.loads(r["structured_prefs"]),
                    "message_count": r["message_count"],
                    "created_at": r["created_at"],
                } for r in rows]
            finally:
                conn.close()
