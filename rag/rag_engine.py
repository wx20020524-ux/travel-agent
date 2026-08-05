"""
RAG 知识库引擎 —— 文档加载、切分、向量化入库、相似度检索。

核心能力:
  1. load_documents()     : 从 knowledge/ 目录批量加载 .md/.txt 文件
  2. split_documents()    : 按 chunk_size=500 + overlap=80 切分文本块
  3. build_vectorstore()  : 调用 Embedding 向量化，FAISS 本地持久化
  4. search_knowledge()   : 相似度检索 Top-K，返回格式化结果给 Agent
  5. add_document()       : 运行时动态添加单篇文档
  6. get_stats()          : 获取知识库状态（文档数、chunk数等）
"""

import os
import glob
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from config import CONFIG


SUPPORTED_EXTS = [".md", ".txt", ".markdown"]


@dataclass
class SearchResult:
    """单条检索结果"""
    content: str
    source: str
    score: float

    def to_str(self) -> str:
        score_str = f"[{self.score:.3f}]"
        return f"【来源: {os.path.basename(self.source)} {score_str}】\n{self.content}"


class RagEngine:
    """
    RAG 知识库引擎（单例语义，需通过 build() 初始化）。

    典型用法:
        rag = RagEngine()
        await rag.build()                       # 加载/构建向量库
        results = rag.search_knowledge("杭州西湖怎么玩")
        for r in results:
            print(r.to_str())
    """

    def __init__(self):
        self._embeddings = None
        self._vectorstore = None
        self._chunks_count = 0
        self._docs_count = 0
        self._built = False

    # ==================== 核心构建 ====================

    async def build(self, force_rebuild: bool = False) -> bool:
        """
        构建或加载向量库。

        Args:
            force_rebuild: True=忽略已有本地向量库，重新从文档构建

        Returns:
            True 表示构建成功，False 表示构建失败（功能降级为不启用 RAG）
        """
        # 1. 创建必要目录
        os.makedirs(CONFIG.rag_knowledge_dir, exist_ok=True)
        os.makedirs(CONFIG.rag_vectorstore_dir, exist_ok=True)

        # 2. 初始化 Embedding
        self._embeddings = CONFIG.create_embeddings()
        if self._embeddings is None:
            print("[RAG] Embedding 初始化失败，RAG 功能不可用")
            return False

        # 3. 判定是否已有本地向量库可加载
        index_file = os.path.join(CONFIG.rag_vectorstore_dir, "index.faiss")
        has_local = os.path.exists(index_file)

        if has_local and not force_rebuild:
            try:
                from langchain_community.vectorstores import FAISS
                self._vectorstore = FAISS.load_local(
                    CONFIG.rag_vectorstore_dir,
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
                self._load_stats()
                self._built = True
                print(f"[RAG] 已加载本地向量库: chunks={self._chunks_count}, docs={self._docs_count}")
                return True
            except Exception as e:
                print(f"[RAG] 加载本地向量库失败，将重新构建: {e}")

        # 4. 从文档重新构建
        documents = self._load_documents_from_disk()
        if not documents:
            print("[RAG] knowledge/ 目录无文档，创建空向量库占位")
            try:
                from langchain_core.documents import Document
                from langchain_community.vectorstores import FAISS
                self._vectorstore = FAISS.from_documents(
                    [Document(page_content="(空知识库占位)", metadata={"source": "placeholder"})],
                    self._embeddings,
                )
                self._save_vectorstore()
                self._docs_count = 0
                self._chunks_count = 0
                self._save_stats()
                self._built = True
                return True
            except Exception as e:
                print(f"[RAG] 创建空向量库失败: {e}")
                return False

        # 5. 切分 + 入库
        chunks = self._split_documents(documents)
        self._chunks_count = len(chunks)
        self._docs_count = len(documents)

        try:
            from langchain_community.vectorstores import FAISS
            print(f"[RAG] 开始向量化 {self._chunks_count} 个文本块...")
            self._vectorstore = FAISS.from_documents(chunks, self._embeddings)
            self._save_vectorstore()
            self._save_stats()
            self._built = True
            print(f"[RAG] 构建成功: {self._docs_count} 文档 / {self._chunks_count} 块")
            return True
        except Exception as e:
            print(f"[RAG] 向量化入库失败: {e}")
            return False

    # ==================== 文档加载 & 切分 ====================

    def _load_documents_from_disk(self) -> List[Any]:
        """从 knowledge/ 目录加载所有 .md/.txt 文件"""
        try:
            from langchain_community.document_loaders import TextLoader
            from langchain_core.documents import Document
        except Exception:
            return []

        docs: List[Document] = []
        for ext in SUPPORTED_EXTS:
            pattern = os.path.join(CONFIG.rag_knowledge_dir, f"**/*{ext}")
            for fpath in glob.glob(pattern, recursive=True):
                try:
                    loader = TextLoader(fpath, encoding="utf-8")
                    for doc in loader.load():
                        doc.metadata["source"] = os.path.relpath(
                            fpath, CONFIG.rag_knowledge_dir
                        )
                        docs.append(doc)
                except Exception as e:
                    print(f"[RAG] 加载文档失败 {fpath}: {e}")
        return docs

    def _split_documents(self, documents: List[Any]) -> List[Any]:
        """按 CONFIG.rag_chunk_size / overlap 切分文档"""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except Exception as e:
            print(f"[RAG] 导入 TextSplitter 失败: {e}")
            return documents

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG.rag_chunk_size,
            chunk_overlap=CONFIG.rag_chunk_overlap,
            separators=CONFIG.rag_separators,
            length_function=len,
        )
        return splitter.split_documents(documents)

    # ==================== 持久化 ====================

    def _save_vectorstore(self):
        if self._vectorstore is not None:
            self._vectorstore.save_local(CONFIG.rag_vectorstore_dir)

    def _save_stats(self):
        stats_path = os.path.join(CONFIG.rag_vectorstore_dir, "stats.json")
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump({
                    "docs_count": self._docs_count,
                    "chunks_count": self._chunks_count,
                    "chunk_size": CONFIG.rag_chunk_size,
                    "chunk_overlap": CONFIG.rag_chunk_overlap,
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_stats(self):
        stats_path = os.path.join(CONFIG.rag_vectorstore_dir, "stats.json")
        try:
            if os.path.exists(stats_path):
                with open(stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                self._docs_count = stats.get("docs_count", 0)
                self._chunks_count = stats.get("chunks_count", 0)
        except Exception:
            self._docs_count = 0
            self._chunks_count = 0

    # ==================== 检索接口 ====================

    def search_knowledge(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        相似度检索知识库。

        Args:
            query: 用户自然语言查询
            top_k: 返回数量，默认取 CONFIG.rag_top_k

        Returns:
            按相似度从高到低排序的 SearchResult 列表（空库或出错时返回空列表）
        """
        if not self._built or self._vectorstore is None or self._docs_count == 0:
            return []

        k = top_k if top_k and top_k > 0 else CONFIG.rag_top_k

        try:
            retriever = self._vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": k,
                    "score_threshold": CONFIG.rag_score_threshold,
                },
            )
            docs = retriever.invoke(query)

            results: List[SearchResult] = []
            for doc in docs:
                content = doc.page_content.strip()
                if not content:
                    continue
                source = doc.metadata.get("source", "unknown")
                score = float(doc.metadata.get("score", 0.0))
                results.append(SearchResult(
                    content=content, source=source, score=score
                ))
            return results
        except Exception as e:
            print(f"[RAG] 检索失败: {e}")
            return []

    def search_knowledge_formatted(self, query: str, top_k: Optional[int] = None) -> str:
        """
        给 Agent 调用的格式化检索结果（纯字符串，便于注入上下文）。
        """
        results = self.search_knowledge(query, top_k)
        if not results:
            return "(知识库无相关内容)"

        lines = ["【知识库检索结果】"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n--- 相关文档 {i} ---")
            lines.append(r.to_str())
        return "\n".join(lines)

    # ==================== 运行时动态添加文档 ====================

    async def add_document(self, content: str, source_name: str = "runtime.md") -> bool:
        """
        动态添加单篇文档到知识库并持久化。

        Args:
            content: 文档正文（字符串）
            source_name: 来源名（仅用于展示，建议加 .md/.txt 后缀）

        Returns:
            True 成功
        """
        if not self._built or self._vectorstore is None:
            await self.build(force_rebuild=False)
            if self._vectorstore is None:
                return False

        try:
            from langchain_core.documents import Document
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            doc = Document(page_content=content, metadata={"source": source_name})
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CONFIG.rag_chunk_size,
                chunk_overlap=CONFIG.rag_chunk_overlap,
                separators=CONFIG.rag_separators,
            )
            chunks = splitter.split_documents([doc])
            self._vectorstore.add_documents(chunks)
            self._chunks_count += len(chunks)
            self._docs_count += 1
            self._save_vectorstore()
            self._save_stats()
            print(f"[RAG] 已新增文档 {source_name}: +{len(chunks)} chunks")
            return True
        except Exception as e:
            print(f"[RAG] 动态新增文档失败: {e}")
            return False

    # ==================== 状态查询 ====================

    def is_ready(self) -> bool:
        return self._built and self._vectorstore is not None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "ready": self._built,
            "docs_count": self._docs_count,
            "chunks_count": self._chunks_count,
            "chunk_size": CONFIG.rag_chunk_size,
            "chunk_overlap": CONFIG.rag_chunk_overlap,
            "top_k": CONFIG.rag_top_k,
            "score_threshold": CONFIG.rag_score_threshold,
            "embedding_model": CONFIG.rag_embedding_model,
            "vectorstore_dir": CONFIG.rag_vectorstore_dir,
            "knowledge_dir": CONFIG.rag_knowledge_dir,
        }


# 模块级单例，全局共享同一个 RAG 引擎
_RAG_SINGLETON: Optional[RagEngine] = None


def get_rag_engine() -> RagEngine:
    """获取 RAG 引擎单例"""
    global _RAG_SINGLETON
    if _RAG_SINGLETON is None:
        _RAG_SINGLETON = RagEngine()
    return _RAG_SINGLETON
