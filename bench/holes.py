"""The typed LLM holes used in the experiments.

Each hole's instruction is the *task contract* for one decision. For routing
holes the contract is written only from information the compiler legitimately
has: the class names and the names of the intents that appear in training
groups. Held-out (cal / test) groups are never named, so the LLM gets no more
knowledge of them than the fast models do; it has to generalise from the
contract plus what it already knows about the world.
"""
from __future__ import annotations

from . import data
from .llm import Hole

NONE = "none_of_these"


def _pretty(s: str) -> str:
    return s.replace("_", " ")


def routing_hole(frame, name: str, what: str, policy_note: str = "", shots: int = 0,
                 none_examples=(), seed: int = 0) -> Hole:
    """Coarse router with a 'none of these' exit.

    The contract lists training-visible request types only. With `shots` > 0 it also quotes
    that many real requests per training group, taken from the *train* split (never from
    calibration or test rows), and `none_examples` quotes human-written out-of-scope requests
    from CLINC's own oos-train partition. A long contract is cheap at run time because the
    whole instruction is a cached prefix.
    """
    train = frame[(frame.role == "train") & (frame.split == "train")]
    lines = []
    classes = sorted(frame.label.unique())
    for c in classes:
        seen = sorted(train.loc[train.label == c, "group"].unique())
        lines.append(f"- {c}: e.g. " + ", ".join(_pretty(g) for g in seen))
    examples = ""
    if shots:
        ex = []
        for c in classes:
            sub = train[train.label == c]
            for g in sorted(sub.group.unique()):
                for t in sub[sub.group == g].sample(shots, random_state=seed).text:
                    ex.append(f'"{t}" -> {c}')
        for t in none_examples:
            ex.append(f'"{t}" -> {NONE}')
        examples = "Examples:\n" + "\n".join(ex) + "\n"
    instruction = (
        f"You route {what} to exactly one category.\n"
        "Categories, each with examples of request types it already handles (it also handles other, "
        "similar request types that are not listed):\n" + "\n".join(lines) + "\n"
        f"- {NONE}: the request does not belong to any category above.\n"
        + examples
        + (policy_note + "\n" if policy_note else "") +
        "Reply with only the category name, nothing else."
    )
    return Hole(name, instruction, classes, none_label=NONE, max_tokens=8)


def clinc_routing_hole(policy_note: str = "", name: str = "clinc_route", shots: int = 1, n_none: int = 20) -> Hole:
    """The CLINC router contract. With a `policy_note` the contract is the *changed* one: the
    training-visible request types named by the policy are listed (and their quoted examples
    labelled) under the new category, and the note itself is added. Held-out groups affected by
    the same policy (e.g. new_card) are still never named; the model has to apply the note."""
    oos = data.clinc_oos()
    none_ex = list(oos[oos.part == "train"].sort_values("uid").text[:n_none]) if shots else ()
    frame = data.clinc_routing()
    if policy_note:
        from .pools import POLICY_GROUPS, POLICY_TO
        frame = frame.copy()
        frame.loc[frame.group.isin(POLICY_GROUPS) & (frame.role == "train"), "label"] = POLICY_TO
    return routing_hole(frame, name, "a user's request to a virtual assistant", policy_note,
                        shots=shots, none_examples=none_ex)


def massive_routing_hole(shots: int = 1) -> Hole:
    return routing_hole(data.massive_routing(), "massive_route", "a user's request to a voice assistant",
                        shots=shots)


def fine_hole(name: str) -> Hole:
    """Fine-grained intent hole: the closed set is the full intent list (names only)."""
    frame = data.fine_frame(name)
    classes = sorted(frame.label.unique())
    what = {"clinc150": "a user's request to a virtual assistant",
            "banking77": "a customer's message to a bank's support chat",
            "massive60": "a user's request to a voice assistant"}[name]
    instruction = (
        f"Classify {what} into exactly one intent from this list:\n" + ", ".join(classes) + "\n"
        f"If none of the intents fits, reply {NONE}.\n"
        "Reply with only the intent name exactly as written in the list, nothing else."
    )
    return Hole(f"{name}_intent", instruction, classes, none_label=NONE, max_tokens=14)


def verify_hole(name: str = "clinc_verify") -> Hole:
    """Yes/no verifier: is the proposed category right for this request?

    The query passed to this hole is '<request>\nProposed category: <label>'. The contract is
    the short one (category names with training-visible request types, no quoted examples), so
    a verification call has a shorter prefix than a classification call.
    """
    base = clinc_routing_hole(shots=0)
    cats = base.instruction.split("Reply with only")[0]
    cats = cats.replace("You route a user's request to a virtual assistant to exactly one category.",
                        "A fast classifier routes a user's request to a virtual assistant to one category.")
    instruction = (cats + "You are checking the fast classifier. Given a request and the category it proposed, "
                   "decide whether the proposed category is the correct one.\nReply with only yes or no.")
    return Hole(name, instruction, ["yes", "no"], query_prefix="Request: ", max_tokens=3)


def verify_query(text: str, proposed: str) -> str:
    return f"{text}\nProposed category: {proposed}"


# The policy change used in the drift simulation (exp6): the business decides that
# everything about physical cards moves from credit_cards to banking. Named groups are
# training-visible; the change is a change to the contract, not to the data.
POLICY_NOTE = ("Policy update: any request about replacing, ordering, receiving or activating a physical "
               "card (for example a damaged card, a replacement card, or when a new card will arrive) "
               "now goes to banking, not credit_cards.")
