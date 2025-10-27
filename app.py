# app.py
import streamlit as st
from typing import List, Dict, Tuple
from collections import namedtuple
import math
import json
from datetime import datetime

st.set_page_config(page_title="TARA — Archetype Matcher", layout="centered")

# ---------- Foundation: Trait tuple and Archetypes (8-d vectors) ----------
TRAITS = ['Introversion', 'Extroversion', 'Planning', 'Spontaneous',
          'Thinking', 'Feeling', 'Intuitive', 'Sensing']
TraitVec = namedtuple('TraitVec', TRAITS)

ARCHETYPES = {
    "The Architect":   TraitVec(0.9, 0.1, 0.95, 0.05, 0.9, 0.1, 0.95, 0.05),
    "The Visionary":   TraitVec(0.8, 0.2, 0.4, 0.6, 0.7, 0.3, 0.9, 0.1),
    "The Innovator":   TraitVec(0.3, 0.7, 0.2, 0.8, 0.8, 0.2, 0.85, 0.15),
    "The Strategist":  TraitVec(0.6, 0.4, 0.9, 0.1, 0.95, 0.05, 0.4, 0.6),
    "The Luminary":    TraitVec(0.9, 0.1, 0.6, 0.4, 0.5, 0.5, 0.9, 0.1),

    "The Empath":      TraitVec(0.8, 0.2, 0.4, 0.6, 0.05, 0.95, 0.85, 0.15),
    "The Connector":   TraitVec(0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6),
    "The Champion":    TraitVec(0.2, 0.8, 0.5, 0.5, 0.3, 0.7, 0.6, 0.4),
    "The Host":        TraitVec(0.4, 0.6, 0.8, 0.2, 0.3, 0.7, 0.2, 0.8),
    "The Muse":        TraitVec(0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.9, 0.1),

    "The Explorer":    TraitVec(0.2, 0.8, 0.1, 0.9, 0.4, 0.6, 0.2, 0.8),
    "The Trailblazer": TraitVec(0.3, 0.7, 0.7, 0.3, 0.8, 0.2, 0.4, 0.6),
    "The Catalyst":    TraitVec(0.1, 0.9, 0.2, 0.8, 0.5, 0.5, 0.3, 0.7),
    "The Artisan":     TraitVec(0.7, 0.3, 0.5, 0.5, 0.6, 0.4, 0.1, 0.9),
    "The Maverick":    TraitVec(0.5, 0.5, 0.1, 0.9, 0.7, 0.3, 0.5, 0.5),

    "The Guardian":    TraitVec(0.8, 0.2, 0.9, 0.1, 0.4, 0.6, 0.2, 0.8),
    "The Anchor":      TraitVec(0.95, 0.05, 0.8, 0.2, 0.5, 0.5, 0.3, 0.7),
    "The Storyteller": TraitVec(0.7, 0.3, 0.6, 0.4, 0.3, 0.7, 0.6, 0.4),
    "The Healer":      TraitVec(0.9, 0.1, 0.5, 0.5, 0.1, 0.9, 0.7, 0.3),
    "The Purist":      TraitVec(0.8, 0.2, 0.9, 0.1, 0.6, 0.4, 0.8, 0.2),
}

# ---------- Strong pairs for direct boosts ----------
STRONG_PAIRS = {
    frozenset(["The Architect", "The Muse"]): 0.94,
    frozenset(["The Visionary", "The Trailblazer"]): 0.95,
    frozenset(["The Innovator", "The Explorer"]): 0.98,
    frozenset(["The Luminary", "The Healer"]): 0.94,
    frozenset(["The Empath", "The Guardian"]): 0.92,
    frozenset(["The Connector", "The Catalyst"]): 0.96,
    frozenset(["The Champion", "The Strategist"]): 0.93,
    frozenset(["The Host", "The Anchor"]): 0.91,
    frozenset(["The Muse", "The Architect"]): 0.94,
    frozenset(["The Storyteller", "The Visionary"]): 0.90,
    frozenset(["The Artisan", "The Purist"]): 0.89,
    frozenset(["The Maverick", "The Explorer"]): 0.92,
    frozenset(["The Catalyst", "The Innovator"]): 0.93,
}

# ---------- Enhanced compatibility logic (kept from your base) ----------
def calculate_compatibility(v1: TraitVec, v2: TraitVec, name1: str, name2: str) -> float:
    pair_key = frozenset([name1, name2])
    if pair_key in STRONG_PAIRS:
        return STRONG_PAIRS[pair_key]

    # Euclidean distance on 8-d trait vectors
    dist = math.dist(v1, v2)
    base_score = 1 / (1 + dist * 1.2)

    # Complementarity bonus — check intro/extro and planning/spontaneous balance
    intro_extra_balance = abs(v1.Introversion - v2.Extroversion) < 0.3
    plan_spont_balance = abs(v1.Planning - v2.Spontaneous) < 0.3

    bonus = 0.0
    if intro_extra_balance:
        bonus += 0.05
    if plan_spont_balance:
        bonus += 0.05

    final_score = max(0.55, min(0.99, base_score + bonus))
    return round(final_score, 4)

# ---------- Precompute compatibility matrix ----------
COMPATIBILITY_MATRIX: Dict[str, Dict[str, float]] = {}
for a1, v1 in ARCHETYPES.items():
    COMPATIBILITY_MATRIX[a1] = {}
    for a2, v2 in ARCHETYPES.items():
        if a1 == a2:
            COMPATIBILITY_MATRIX[a1][a2] = 0.85
        else:
            COMPATIBILITY_MATRIX[a1][a2] = calculate_compatibility(v1, v2, a1, a2)

# ---------- Quiz questions (16) - single-axis mapping ----------
# Each question targets one core axis: E_I, N_S, T_F, J_P
QUESTIONS = [
  {"id":"Q1","prompt":"It's your first date—are you:","axis":"E_I","choices":["Breaking the ice with jokes","Letting them do most of the talking"],"first_maps_to":"E"},
  {"id":"Q2","prompt":"Meeting your date's squad, you:","axis":"E_I","choices":["Chat up everyone in sight","Stick close to your date"],"first_maps_to":"E"},
  {"id":"Q3","prompt":"Weekend with bae—after, you:","axis":"E_I","choices":["Still up for another social hang","Craving solo chill time"],"first_maps_to":"E"},
  {"id":"Q4","prompt":"Dreaming up your future together, you:","axis":"N_S","choices":["Picture wild adventures ahead","See cozy routines and actual plans"],"first_maps_to":"N"},
  {"id":"Q5","prompt":"Your crush is:","axis":"N_S","choices":["Big on wild ideas","Solid and always practical"],"first_maps_to":"N"},
  {"id":"Q6","prompt":"Date night plans are usually:","axis":"N_S","choices":["Unusual experiences, why not?","The usual fav spot, guaranteed vibes"],"first_maps_to":"N"},
  {"id":"Q7","prompt":"If your partner forgets your birthday, you:","axis":"T_F","choices":["Laugh it off, mistakes happen","Low-key hurt but play it cool"],"first_maps_to":"T"},
  {"id":"Q8","prompt":"You and your boo disagree—you:","axis":"T_F","choices":["Stay chill, focus on the facts","Talk about feelings and vibes first"],"first_maps_to":"T"},
  {"id":"Q9","prompt":"When arguing, you want your date to:","axis":"T_F","choices":["Just fix the problem","Notice your mood first"],"first_maps_to":"T"},
  {"id":"Q10","prompt":"Talking future goals with your significant other, you:","axis":"J_P","choices":["Text out a plan tonight","Figure it out as you go"],"first_maps_to":"J"},
  {"id":"Q11","prompt":"Planning a road trip, you:","axis":"J_P","choices":["Share the Google map week before","Wing it, set off when you feel like"],"first_maps_to":"J"},
  {"id":"Q12","prompt":"Date cancels suddenly, you’re:","axis":"J_P","choices":["A little salty about the mix-up","No worries, plans change"],"first_maps_to":"J"},
  {"id":"Q13","prompt":"Meeting bae's fam, you’re:","axis":"E_I","choices":["Chatty from the jump","Hanging back until you vibe"],"first_maps_to":"E"},
  {"id":"Q14","prompt":"Dream relationship is about:","axis":"N_S","choices":["Growth, sharing goals, big vision","Small moments, steady routines"],"first_maps_to":"N"},
  {"id":"Q15","prompt":"Date asks for advice, you:","axis":"T_F","choices":["Drop logical tips","Share some hype and support"],"first_maps_to":"T"},
  {"id":"Q16","prompt":"Future talk with your significant other—you:","axis":"J_P","choices":["Like to have goals and timelines","See where things go"],"first_maps_to":"J"}
]

# ---------- Scoring helpers (0.5 ± step per axis) ----------
AXIS_POSITIVE = {"E_I": "E", "N_S": "N", "T_F": "T", "J_P": "J"}
PAIRS = {"E_I": ("E","I"), "N_S": ("N","S"), "T_F": ("T","F"), "J_P": ("J","P")}

def prepare_steps(sampled_questions: List[dict]) -> Dict[str, float]:
    counts = {"E":0,"N":0,"T":0,"J":0}
    for q in sampled_questions:
        pos = AXIS_POSITIVE[q["axis"]]
        counts[pos] += 1
    steps = {k: (0.5 / counts[k]) if counts[k] > 0 else 0.0 for k in counts}
    return steps

def score_axes(sampled_questions: List[dict], chosen_indices: List[int]) -> Dict[str, float]:
    """
    Returns axis dict with E,N,T,J each in [0,1].
    Start at 0.5 and apply +step/-step according to choice and first_maps_to.
    """
    axis = {"E":0.5,"N":0.5,"T":0.5,"J":0.5}
    steps = prepare_steps(sampled_questions)
    for q, opt_idx in zip(sampled_questions, chosen_indices):
        pos = AXIS_POSITIVE[q["axis"]]
        step = steps[pos]
        first_maps_to = q.get("first_maps_to", q["axis"].split("_")[0])
        if first_maps_to == pos:
            delta = step if opt_idx == 0 else -step
        else:
            delta = -step if opt_idx == 0 else step
        axis[pos] = max(0.0, min(1.0, axis[pos] + delta))
    return axis

def axes_to_traitvec(axis: Dict[str,float]) -> TraitVec:
    """
    Convert E/N/T/J axis values into 8-d TraitVec:
    Introversion = 1 - E, Extroversion = E,
    Planning = J, Spontaneous = 1 - J,
    Thinking = T, Feeling = 1 - T,
    Intuitive = N, Sensing = 1 - N
    """
    E = axis["E"]; N = axis["N"]; T = axis["T"]; J = axis["J"]
    return TraitVec(
        1.0 - E,   # Introversion
        E,         # Extroversion
        J,         # Planning
        1.0 - J,   # Spontaneous
        T,         # Thinking
        1.0 - T,   # Feeling
        N,         # Intuitive
        1.0 - N    # Sensing
    )

def axis_to_mbti(axis: Dict[str,float]) -> str:
    return (
        ("E" if axis["E"] >= 0.5 else "I") +
        ("N" if axis["N"] >= 0.5 else "S") +
        ("T" if axis["T"] >= 0.5 else "F") +
        ("J" if axis["J"] >= 0.5 else "P")
    )

# ---------- Matching helpers ----------
def rank_archetypes_for_user(user_vec: TraitVec) -> List[Tuple[str, float]]:
    """
    Score user's 8-d TraitVec against every archetype using calculate_compatibility,
    return sorted list of (archetype_name, score) descending.
    """
    scores = []
    for name, arche_vec in ARCHETYPES.items():
        sc = calculate_compatibility(user_vec, arche_vec, "User", name)
        scores.append((name, sc))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# ---------- Streamlit UI ----------
st.title("TARA — Relationship Archetype Matcher")
st.markdown("Answer the quick relationship-style quiz. Results show your MBTI-like axes and closest archetypes.")

with st.form("quiz_form"):
    st.markdown("### Questions")
    cols = st.columns([1, 1])
    # two-column compact layout
    for i, q in enumerate(QUESTIONS):
        col = cols[i % 2]
        with col:
            st.radio(q["prompt"], q["choices"], key=q["id"], index=0)
    submitted = st.form_submit_button("See my match")

if submitted:
    # collect answers
    choices = []
    for q in QUESTIONS:
        sel = st.session_state.get(q["id"], q["choices"][0])
        choices.append(q["choices"].index(sel))

    # compute axis -> traitvec -> archetypes
    axis = score_axes(QUESTIONS, choices)
    mbti = axis_to_mbti(axis)
    user_traitvec = axes_to_traitvec(axis)
    ranked = rank_archetypes_for_user(user_traitvec)

    # Display results
    st.subheader("🎯 Your Results")
    st.markdown(f"**MBTI-style:** `{mbti}`")
    st.markdown("**Axis scores:**")
    st.write(f"- Extraversion (E): **{axis['E']:.3f}**")
    st.write(f"- Intuition (N): **{axis['N']:.3f}**")
    st.write(f"- Thinking (T): **{axis['T']:.3f}**")
    st.write(f"- Judging (J): **{axis['J']:.3f}**")

    st.markdown("**Derived trait vector (8-d):**")
    tv = user_traitvec
    st.write({
        "Introversion": round(tv.Introversion, 3),
        "Extroversion": round(tv.Extroversion, 3),
        "Planning": round(tv.Planning, 3),
        "Spontaneous": round(tv.Spontaneous, 3),
        "Thinking": round(tv.Thinking, 3),
        "Feeling": round(tv.Feeling, 3),
        "Intuitive": round(tv.Intuitive, 3),
        "Sensing": round(tv.Sensing, 3),
    })

    best_name, best_score = ranked[0]
    st.markdown(f"## 🏷️ Closest Archetype: **{best_name}** — **{best_score:.2%}**")
    st.markdown("### Top 10 Matches")
    for rank, (name, score) in enumerate(ranked[:10], start=1):
        st.write(f"{rank}. **{name}** — {score:.2%}")

    # show raw compatibility slice (optionally)
    if st.checkbox("Show full compatibility matrix snapshot"):
        st.json({k: COMPATIBILITY_MATRIX.get(k) for k in list(COMPATIBILITY_MATRIX.keys())[:5]})

    # download
    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "choices": choices,
        "axis": axis,
        "mbti": mbti,
        "best_archetype": best_name,
        "best_score": best_score,
    }
    st.download_button("⬇️ Download Results (JSON)", json.dumps(out, indent=2), file_name=f"tara_result_{mbti}.json", mime="application/json")

st.markdown("---")
st.markdown("<small>Tip: you can stratify sampling (deliver 15 of 50) later — this runs on the full 16 items for demo purposes.</small>", unsafe_allow_html=True)
