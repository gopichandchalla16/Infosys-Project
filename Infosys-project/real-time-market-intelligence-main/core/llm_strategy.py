# core/llm_strategy.py

def generate_strategic_explanation(
    company: str,
    competitive_index: float,
    strategic_signal: dict,
    sentiment_counts: dict,
):
    """
    Generates an executive-style strategic explanation.
    This function is LLM-ready (OpenAI / LLaMA / local LLM).
    Currently implemented as a deterministic LLM-style response
    for internship safety and reproducibility.
    """

    pos = sentiment_counts.get("positive", 0)
    neg = sentiment_counts.get("negative", 0)
    neu = sentiment_counts.get("neutral", 0)

    signal = strategic_signal.get("signal", "MONITOR")
    confidence = strategic_signal.get("confidence", "Medium")

    if signal == "OPPORTUNITY":
        explanation = f"""
{company} is currently exhibiting strong strategic momentum.
The Competitive Positioning Index of {competitive_index}/100
indicates favorable market conditions supported by positive
sentiment ({pos} positive vs {neg} negative news items).

Upward price movement combined with a positive forecast trend
suggests potential growth opportunities. This environment is
conducive for expansion, investment consideration, or strategic
initiatives aligned with market confidence.

Recommended Action:
• Monitor short-term price movements
• Capitalize on favorable sentiment trends
• Track competitor reactions closely
"""
    elif signal == "THREAT":
        explanation = f"""
{company} is facing elevated strategic risk signals.
A Competitive Positioning Index of {competitive_index}/100,
along with dominant negative sentiment ({neg} negative articles),
indicates potential downside pressure.

Declining momentum and unfavorable forecast indicators suggest
the need for cautious strategic planning. External factors such
as regulatory developments or competitive pressures may be
influencing market perception.

Recommended Action:
• Strengthen risk monitoring
• Delay aggressive strategic commitments
• Reassess exposure and mitigation plans
"""
    else:
        explanation = f"""
{company} currently presents mixed strategic signals.
With a Competitive Positioning Index of {competitive_index}/100,
market sentiment remains balanced ({pos} positive, {neu} neutral,
{neg} negative).

Forecast trends and price momentum do not indicate a clear
direction at this stage. Strategic patience and continuous
monitoring are advised until clearer signals emerge.

Recommended Action:
• Maintain observation posture
• Monitor sentiment shifts
• Await confirmation from market trends
"""

    return explanation.strip()
