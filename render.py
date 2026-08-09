"""渲染工具 —— JSON 解析、CLI 格式化、Streamlit 组件。"""
import json


def parse_plan(text: str) -> dict | None:
    """从混合文本中提取并解析旅行计划 JSON。"""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        return json.loads(text[start:end])
    except (json.JSONDecodeError, KeyError):
        return None


# ==================== CLI 格式化 ====================

def _weather_icon(weather: str) -> str:
    """天气文字 -> 纯文本图标（兼容 Windows GBK）"""
    mapping = {
        "晴": "[Sun]", "多云": "[Cloudy]", "阴": "[Overcast]",
        "小雨": "[Rain]", "中雨": "[Rain]", "大雨": "[HeavyRain]", "暴雨": "[Storm]",
        "雪": "[Snow]", "雾": "[Fog]", "霾": "[Haze]",
    }
    for key, icon in mapping.items():
        if key in weather:
            return icon
    return "[?]"


def format_plan_cli(json_text: str) -> str | None:
    """将 Planner JSON 渲染为 CLI 可读的中文旅行计划。"""
    data = parse_plan(json_text)
    if data is None:
        return None

    lines = []
    city = data.get("city", "未知")
    start_date = data.get("start_date", "")
    end_date = data.get("end_date", "")

    lines.append("")
    lines.append("=" * 60)
    title = f"  {city} {start_date} ~ {end_date} 旅行计划"
    lines.append(f"  {title}")
    lines.append("=" * 60)

    # 天气
    weather_info = data.get("weather_info", [])
    if weather_info:
        lines.append("")
        lines.append("[Weather] 天气概况")
        for w in weather_info:
            d = w.get("date", "")[-5:]
            di = _weather_icon(w.get("day_weather", ""))
            ni = _weather_icon(w.get("night_weather", ""))
            lines.append(
                f"   {d}  {di} {w.get('day_weather', '?')} -> "
                f"{ni} {w.get('night_weather', '?')}  "
                f"{w.get('day_temp', '?')}C / {w.get('night_temp', '?')}C  "
                f"{w.get('wind_direction', '')}{w.get('wind_power', '')}"
            )

    # 每日行程
    for day in data.get("days", []):
        idx = day.get("day_index", 0) + 1
        d = day.get("date", "")[-5:]
        desc = day.get("description", "")
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"Day {idx}  {d}  {desc}")
        lines.append("-" * 60)

        hotel = day.get("hotel", {}) or {}
        if hotel.get("name"):
            lines.append(
                f"  [Hotel] {hotel['name']}  *{hotel.get('rating', '')}  "
                f"Y{hotel.get('estimated_cost', 0)}/晚  |  {hotel.get('address', '')}"
            )
        lines.append(f"  [Transit] {day.get('transportation', '')}")

        attractions = day.get("attractions", [])
        if attractions:
            lines.append(f"  [Attraction] 景点 ({len(attractions)}个):")
            for a in attractions:
                ticket = a.get("ticket_price", 0)
                ts = "免费" if ticket == 0 else f"Y{ticket}"
                lines.append(f"     | {a.get('name', '?')}")
                lines.append(
                    f"       {a.get('address', '')}  |  {a.get('category', '')}  |  "
                    f"游玩约{a.get('visit_duration', 0)}分钟  |  {ts}"
                )

        meals = day.get("meals", [])
        if meals:
            lines.append("  [Food] 餐饮:")
            for m in meals:
                mt = {"breakfast": "早", "lunch": "午", "dinner": "晚"}
                label = mt.get(m.get("type", ""), "餐")
                lines.append(f"     {label} {m.get('name', '?')}  Y{m.get('estimated_cost', 0)}")

    # 预算
    budget = data.get("budget", {})
    if budget:
        lines.append("")
        lines.append("-" * 60)
        lines.append("[Budget] 预算汇总")
        lines.append(
            f"   景点: Y{budget.get('total_attractions', 0):>6}  |  "
            f"酒店: Y{budget.get('total_hotels', 0):>6}  |  "
            f"餐饮: Y{budget.get('total_meals', 0):>6}  |  "
            f"交通: Y{budget.get('total_transportation', 0):>6}"
        )
        lines.append(f"   总计: Y{budget.get('total', 0):,}")

    # 建议
    suggestions = data.get("overall_suggestions", "")
    if suggestions:
        lines.append("")
        lines.append("[Tips] 旅行建议")
        for tip in suggestions.replace("；", ";").split(";"):
            tip = tip.strip()
            if tip:
                lines.append(f"   {tip}")

    lines.append("")
    return "\n".join(lines)
