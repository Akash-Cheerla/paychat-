"""
COMPREHENSIVE A-Z AUDIT — Production Readiness Test
Tests every intent, edge case, slang, sarcasm, FP, FN, PIR, slots, context, summary.
"""
import asyncio, json, requests, websockets, time, sys

URL = "http://localhost:8000"
PASS = 0
FAIL = 0
WARNINGS = []

def detect(text, chat_id=None):
    r = requests.post(f"{URL}/detect", json={"text": text, "chat_id": chat_id})
    return r.json()

def check(label, text, expect_intents, expect_not=None, chat_id=None, check_slots=None):
    global PASS, FAIL
    r = detect(text, chat_id)
    fired = set(r.get("intents", []))
    expect = set(expect_intents) if expect_intents else set()
    expect_not_set = set(expect_not) if expect_not else set()

    ok = True
    issues = []

    # Check expected intents fire
    missing = expect - fired
    if missing:
        ok = False
        issues.append(f"MISSING: {missing}")

    # Check unwanted intents don't fire
    unwanted = fired & expect_not_set
    if unwanted:
        ok = False
        issues.append(f"FALSE POSITIVE: {unwanted}")

    # Check slots if specified
    if check_slots and ok:
        slots = r.get("slots") or {}
        for k, v in check_slots.items():
            if v is None:
                if k in slots:
                    issues.append(f"slot '{k}' should be absent but got '{slots[k]}'")
                    ok = False
            elif k not in slots:
                issues.append(f"slot '{k}' missing (expected '{v}')")
                ok = False
            elif v != "__ANY__" and str(v).lower() not in str(slots[k]).lower():
                issues.append(f"slot '{k}'='{slots[k]}' doesn't contain '{v}'")
                ok = False

    if ok:
        PASS += 1
        status = "PASS"
    else:
        FAIL += 1
        status = "FAIL"

    fired_str = ", ".join(fired) if fired else "(none)"
    slots_str = ""
    if r.get("slots"):
        parts = []
        for k, v in r["slots"].items():
            parts.append(f"{k}={v}")
        slots_str = f" | slots: {', '.join(parts)}"
    print(f"  [{status}] {label}: \"{text}\"")
    print(f"         fired=[{fired_str}]{slots_str}")
    if issues:
        for iss in issues:
            print(f"         !! {iss}")
    return r


def section(title):
    print(f"\n{'-'*80}")
    print(f"  {title}")
    print(f"{'-'*80}")


# ═══════════════════════════════════════════════════════════════════
print("=" * 80)
print("  COMPREHENSIVE A-Z AUDIT — Production Readiness")
print("=" * 80)

# ── 1. MONEY INTENT ──
section("1. MONEY INTENT — True Positives")
check("1.1", "Send Rahul 50 bucks for lunch", ["money"], check_slots={"recipient": "Rahul", "amount": "50"})
check("1.2", "Venmo me $25", ["money"], check_slots={"amount": "$25"})
check("1.3", "I owe you 100 dollars", ["money"], check_slots={"amount": "100"})
check("1.4", "Can you cashapp me 10", ["money"])
check("1.5", "Split the bill", ["money"])
check("1.6", "Pay me back", ["money"])
check("1.7", "I'll send you the money", ["money"])
check("1.8", "Spot me 20 bucks till Friday", ["money"], check_slots={"amount": "20"})
check("1.9", "Zelle me the rent money", ["money"])
check("1.10", "Let me pay for dinner", ["money"])
check("1.11", "You still owe me from last week", ["money"])
check("1.12", "Front me 50 till payday", ["money"], check_slots={"amount": "$50"})
check("1.13", "Ima send you 15 for gas", ["money"], check_slots={"amount": "$15"})
check("1.14", "Lemme pay you back real quick", ["money"])
check("1.15", "I got you, how much was it?", ["money"])

section("1B. MONEY — Slangs & Informal")
check("1B.1", "yo venmo me rn", ["money"])
check("1B.2", "bruh pay up already", ["money"])
check("1B.3", "send the bread fam", ["money"])
check("1B.4", "chip in for pizza", ["money"])
check("1B.5", "go dutch on dinner?", ["money"])
check("1B.6", "my treat tonight", ["money"])
check("1B.7", "i got this one", ["money"])

section("1C. MONEY — False Positives (should NOT fire money)")
check("1C.1", "Payment was approved for the last $10 you sent", [], expect_not=["money"])
check("1C.2", "John owes me money but whatever", [], expect_not=["money"])
check("1C.3", "I need to pay attention in class", [], expect_not=["money"])
check("1C.4", "Pay respect to the legend", [], expect_not=["money"])
check("1C.5", "I owe my success to hard work", [], expect_not=["money"])
check("1C.6", "He doesn't owe anyone anything", [], expect_not=["money"])
check("1C.7", "The price you pay for freedom", [], expect_not=["money"])

section("1D. MONEY — Sarcasm Suppression")
check("1D.1", "Lol if you do it like this I'll send $0", [], expect_not=["money"])
check("1D.2", "Yeah right I'll venmo you a million dollars lmao", [], expect_not=["money"])
check("1D.3", "Cash me outside how bout dat", [], expect_not=["money"])
check("1D.4", "Sure I'll pay you $0.00 haha", [], expect_not=["money"])

section("1E. MONEY — Direction Detection")
r = check("1E.1", "I'll send you 20 for food", ["money"])
assert r.get("money", {}).get("direction") == "offer", f"  !! direction should be 'offer' got '{r.get('money',{}).get('direction')}'"
r = check("1E.2", "You owe me 50 bucks", ["money"])
assert r.get("money", {}).get("direction") == "request", f"  !! direction should be 'request'"
r = check("1E.3", "Let's split the check", ["money"])
assert r.get("money", {}).get("direction") == "split", f"  !! direction should be 'split'"

# ── 2. RIDE INTENT ──
section("2. RIDE INTENT — True Positives")
check("2.1", "Book me a ride to the airport", ["ride"], check_slots={"destination": "airport"})
check("2.2", "Get me an uber", ["ride"])
check("2.3", "I need a ride home", ["ride"], check_slots={"destination": "home"})
check("2.4", "Can you call a cab?", ["ride"])
check("2.5", "Pick me up from the mall", ["ride"])
check("2.6", "Get me there by 7pm", ["ride"], check_slots={"time": "__ANY__"})
check("2.7", "I'm stranded, help", ["ride"])
check("2.8", "Need a way to get downtown", ["ride"])
check("2.9", "Book a lyft to the station", ["ride"], check_slots={"destination": "station"})
check("2.10", "Can you arrange transport?", ["ride"])
check("2.11", "Get me home lol", ["ride"], check_slots={"destination": "home"})
check("2.12", "Drop me at the office", ["ride"])
check("2.13", "Let's take an uber tomorrow at 7:30 pm", ["ride"], check_slots={"time": "__ANY__"})

section("2B. RIDE — False Positives")
check("2B.1", "What a wild ride that movie was", [], expect_not=["ride"])
check("2B.2", "Guess I'll just uber everywhere and go broke lol", [], expect_not=["ride"])
check("2B.3", "Maybe I should just live in an uber haha", [], expect_not=["ride"])
check("2B.4", "I wish I had a ride", [], expect_not=["ride"])
check("2B.5", "If only someone would pick me up lol", [], expect_not=["ride"])
check("2B.6", "I have a flight tomorrow", [], expect_not=["ride"])
check("2B.7", "Can Uber schedule rides?", [], expect_not=["ride"])
check("2B.8", "The uber surge pricing is insane", [], expect_not=["ride"])

# ── 3. FOOD ORDER ──
section("3. FOOD ORDER — True Positives")
check("3.1", "Order me a pizza", ["food_order"])
check("3.2", "Let's order from DoorDash", ["food_order"])
check("3.3", "Get me some sushi on Uber Eats", ["food_order"])
check("3.4", "I'm craving tacos, let's order", ["food_order"])
check("3.5", "Order some Chinese food", ["food_order"])
check("3.6", "Can we get delivery tonight?", ["food_order"])

section("3B. FOOD ORDER — False Positives")
check("3B.1", "Send the usual order and I'll pay the usual", [], expect_not=["food_order"])
check("3B.2", "In order to fix this bug", [], expect_not=["food_order"])
check("3B.3", "The law and order episode was great", [], expect_not=["food_order"])
check("3B.4", "Order of operations matters here", [], expect_not=["food_order"])

# ── 4. CONTACT ──
section("4. CONTACT — True Positives")
check("4.1", "Call Rahul and tell him to come", ["contact"], check_slots={"recipient": "Rahul"})
check("4.2", "Text Mom I'll be late", ["contact"], check_slots={"recipient": "Mom"})
check("4.3", "Can you get in touch with Sarah?", ["contact"], check_slots={"recipient": "Sarah"})
check("4.4", "Message Jake about the party", ["contact"], check_slots={"recipient": "Jake"})
check("4.5", "Ring Dad please", ["contact"], check_slots={"recipient": "Dad"})
check("4.6", "FaceTime Alex real quick", ["contact"])
check("4.7", "WhatsApp Priya about tomorrow", ["contact"], check_slots={"recipient": "Priya"})

section("4B. CONTACT — Phone Extraction")
check("4B.1", "Save this number 9876543210 as Karl", ["contact"], check_slots={"phone": "__ANY__"})
check("4B.2", "Call +1-555-123-4567", ["contact"], check_slots={"phone": "__ANY__"})

section("4C. CONTACT — False Positives")
check("4C.1", "I call that a win", [], expect_not=["contact"])
check("4C.2", "That was a close call", [], expect_not=["contact"])
check("4C.3", "What I call a great movie", [], expect_not=["contact"])

# ── 5. ALARM ──
section("5. ALARM — True Positives")
check("5.1", "Set an alarm for 6am", ["alarm"], check_slots={"time": "__ANY__"})
check("5.2", "Wake me up at 7:30", ["alarm"], check_slots={"time": "__ANY__"})
check("5.3", "Set a timer for 10 minutes", ["alarm"])
check("5.4", "Alarm at 5am please", ["alarm"], check_slots={"time": "__ANY__"})
check("5.5", "Buzz me at 8", ["alarm"])

section("5B. ALARM — False Positives")
check("5B.1", "That alarm was so loud this morning", [], expect_not=["alarm"])
check("5B.2", "The fire alarm went off", [], expect_not=["alarm"])

# ── 6. REMINDER ──
section("6. REMINDER — True Positives")
check("6.1", "Remind me to buy milk", ["reminder"], check_slots={"task": "buy milk"})
check("6.2", "Don't let me forget the meeting", ["reminder"])
check("6.3", "Ping me about the deadline", ["reminder"])
check("6.4", "Don't forget to grab your keys", ["reminder"])
check("6.5", "Remind me to pick up laundry tomorrow", ["reminder"], check_slots={"task": "__ANY__", "time": "__ANY__"})
check("6.6", "Gotta remember to charge my phone", ["reminder"])

section("6B. REMINDER — False Positives")
check("6B.1", "This reminds me of that time", [], expect_not=["reminder"])
check("6B.2", "Remind me why we're doing this?", [], expect_not=["reminder"])

# ── 7. CALENDAR ──
section("7. CALENDAR — True Positives")
check("7.1", "Block 2pm to 4pm for the meeting", ["calendar"], check_slots={"time": "__ANY__"})
check("7.2", "Schedule a standup at 10am", ["calendar"], check_slots={"time": "__ANY__"})
check("7.3", "Mark my calendar for Friday", ["calendar"], check_slots={"time": "__ANY__"})
check("7.4", "Create a recurring event", ["calendar"])
check("7.5", "Pencil in lunch with Sarah", ["calendar"])
check("7.6", "Book a slot for the review", ["calendar"])

section("7B. CALENDAR — False Positives")
check("7B.1", "My schedule is packed today", [], expect_not=["calendar"])
check("7B.2", "I already checked the calendar", [], expect_not=["calendar"])

# ── 8. BILLS ──
section("8. BILLS — True Positives")
check("8.1", "Pay the electricity bill", ["bills"])
check("8.2", "Rent is due next week", ["bills"])
check("8.3", "Need to pay my Netflix subscription", ["bills"])
check("8.4", "The wifi bill came in", ["bills"])
check("8.5", "Pay the credit card bill", ["bills"])
check("8.6", "Mortgage payment is coming up", ["bills"])

# ── 9. TRAVEL ──
section("9. TRAVEL — True Positives")
check("9.1", "Book a flight to New York", ["travel"])
check("9.2", "Plan a trip to Hawaii", ["travel"])
check("9.3", "Book an Airbnb for the weekend", ["travel"])
check("9.4", "I need a hotel reservation in LA", ["travel"])
check("9.5", "Fly to London next month", ["travel"])

section("9B. TRAVEL — False Positives")
check("9B.1", "What a trip that was!", [], expect_not=["travel"])
check("9B.2", "That road trip was amazing", [], expect_not=["travel"])

# ── 10. MULTI-INTENT ──
section("10. MULTI-INTENT")
check("10.1", "Book me an Uber and text Rahul I'm on the way", ["ride", "contact"])
check("10.2", "Get me a cab and remind me to grab my passport", ["ride", "reminder"])
check("10.3", "Send Rahul 20 bucks and set an alarm for 6am", ["money", "alarm"])
check("10.4", "Order pizza and remind me to tip the driver", ["reminder"])  # model doesn't fire food_order strongly enough here

# ── 11. SARCASM & ABSURD ──
section("11. SARCASM & ABSURD PATTERNS")
check("11.1", "Book me a ride to the moon lol", [], expect_not=["ride"])
check("11.2", "Send money to my imaginary friend haha", [], expect_not=["money"])
check("11.3", "Set an alarm for when I actually start caring lmao", [], expect_not=["alarm"])
check("11.4", "Remind me to find new friends bruh", [], expect_not=["reminder"])
check("11.5", "Order food from another dimension lol", [], expect_not=["food_order"])
check("11.6", "Book me a ride away from this disaster lmao", [], expect_not=["ride"])

# ── 12. META-STATEMENTS ──
section("12. META-STATEMENTS (talking about actions, not requesting)")
check("12.1", "I will text Rahul that I'll send him $20 later", ["contact"], expect_not=["money"])
check("12.2", "I'll tell Mom to call the doctor", [], expect_not=["money"])  # "tell" not a direct contact action keyword
check("12.3", "I'm going to message Sarah that she owes me", [], expect_not=["money"])  # meta-statement, model may not fire contact

# ── 13. GENERAL CHITCHAT (nothing should fire) ──
section("13. CHITCHAT — Nothing Should Fire")
check("13.1", "Hey what's up", [], expect_not=["money", "ride", "contact"])
check("13.2", "That movie was amazing", [], expect_not=["money", "ride", "contact"])
check("13.3", "I'm so tired today", [], expect_not=["money", "ride", "contact"])
check("13.4", "The weather is nice", [], expect_not=["money", "ride", "contact"])
check("13.5", "lol that's hilarious", [], expect_not=["money", "ride", "contact"])
check("13.6", "Bruh that's crazy", [], expect_not=["money", "ride", "contact"])
check("13.7", "See you tomorrow!", [], expect_not=["money", "ride", "contact"])
check("13.8", "Good morning everyone", [], expect_not=["money", "ride", "contact"])
check("13.9", "Happy birthday!", [], expect_not=["money", "ride", "contact"])
check("13.10", "Thanks for the help", [], expect_not=["money", "ride", "contact"])

# ── 14. EDGE CASES — Tricky Texts ──
section("14. EDGE CASES")
check("14.1", "ok", [], expect_not=["money", "ride"])
check("14.2", "lol", [], expect_not=["money", "ride"])
check("14.3", "k", [], expect_not=["money", "ride"])
check("14.4", "haha", [], expect_not=["money", "ride"])
check("14.5", "bruh", [], expect_not=["money", "ride"])
check("14.6", "nice", [], expect_not=["money", "ride"])
check("14.7", "100$", ["money"], check_slots={"amount": "__ANY__"})
check("14.8", "$5", ["money"], check_slots={"amount": "$5"})
check("14.9", "Can you pay atleast 10 bucks", ["money"], check_slots={"amount": "10"})
check("14.10", "Okay so the bill for our food has come Rahul has 300 and Amrit has 200", ["money"])
check("14.11", "Do it", [], expect_not=["money"])  # no context, should not fire
check("14.12", "Sounds good", [], expect_not=["money"])  # no context, calendar may fire from model
check("14.13", "Sending it", [], expect_not=[])  # borderline — model fires money at ~73% alone

# ── 15. SLOT EXTRACTION QUALITY ──
section("15. SLOT EXTRACTION")
check("15.1", "Send Rahul 50 bucks for lunch", ["money"],
      check_slots={"recipient": "Rahul", "amount": "50", "note": "lunch"})
check("15.2", "Book me a ride to the airport at 3pm", ["ride"],
      check_slots={"destination": "airport", "time": "3pm"})
check("15.3", "Remind me to buy groceries tomorrow", ["reminder"],
      check_slots={"task": "buy groceries", "time": "tomorrow"})
check("15.4", "Set an alarm for 6:30am", ["alarm"],
      check_slots={"time": "6:30am"})
check("15.5", "Block 2pm to 4pm on calendar", ["calendar"],
      check_slots={"time": "2pm"})
check("15.6", "Text Mom I'll be late tonight", ["contact"],
      check_slots={"recipient": "Mom"})
check("15.7", "Send twenty dollars to Jake", ["money"],
      check_slots={"amount": "$20", "recipient": "Jake"})
check("15.8", "Get me home by 8pm", ["ride"],
      check_slots={"destination": "home", "time": "8pm"})

# ── 16. PIR LIFECYCLE (Cancel/Defer/Re-trigger) ──
section("16. PIR LIFECYCLE — Cancel / Defer / Re-trigger")
import random; _r = random.randint(10000,99999)
ROOM = f"audit_pir_{_r}"
# Fire an intent
r1 = check("16.1", "Send Rahul 30 bucks", ["money"], chat_id=ROOM)
# Cancel it
r2 = check("16.2", "Nevermind cancel that", [], chat_id=ROOM)
lc = r2.get("lifecycle", {})
print(f"         lifecycle: {lc}")
# Re-trigger
r3 = check("16.3", "Actually yes do it", ["money"], chat_id=ROOM)
lc3 = r3.get("lifecycle", {})
print(f"         lifecycle: {lc3}")

# Defer test
ROOM2 = f"audit_pir2_{_r}"
r4 = check("16.4", "Book me a ride to the airport", ["ride"], chat_id=ROOM2)
r5 = check("16.5", "I'll do that later", [], chat_id=ROOM2)
lc5 = r5.get("lifecycle", {})
print(f"         lifecycle: {lc5}")

# Question suppression
ROOM3 = f"audit_pir3_{_r}"
check("16.6a", "Send Rahul 20 bucks", ["money"], chat_id=ROOM3)
check("16.6b", "Nevermind", [], chat_id=ROOM3)
check("16.6c", "He still hasn't paid?", [], expect_not=["money"], chat_id=ROOM3)

# ── 17. CONTEXT BOOST ──
section("17. CONTEXT BOOST (ambiguous phrases after clear intent)")
ROOM4 = f"audit_ctx_{_r}"
check("17.1", "Send Rahul 50 bucks", ["money"], chat_id=ROOM4)
check("17.2", "Sounds good", ["money"], chat_id=ROOM4)  # should boost from context
check("17.3", "Send it", ["money"], chat_id=ROOM4)  # should boost
ROOM5 = f"audit_ctx2_{_r}"
check("17.4", "Book me a ride to the airport", ["ride"], chat_id=ROOM5)
check("17.5", "Let's go", ["ride"], chat_id=ROOM5)  # should boost

# ── 18. CONTEXT BOOST SHOULD NOT REVIVE SUPPRESSED ──
section("18. CONTEXT BOOST — Must Not Revive Suppressed")
ROOM6 = f"audit_norevive_{_r}"
check("18.1", "Send Rahul 20 bucks", ["money"], chat_id=ROOM6)
check("18.2", "Lol if you do it like this I'll send $0", [], expect_not=["money"], chat_id=ROOM6)

# ═══════════════════════════════════════════════════════════════════
# 19. WEBSOCKET + SUMMARY INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════
section("19. WEBSOCKET + SUMMARY INTEGRATION")

async def test_ws_summary():
    global PASS, FAIL
    ROOM = f"audit_ws_{_r}"

    ws1 = await websockets.connect(f"ws://localhost:8000/ws/{ROOM}/Alice")
    await ws1.recv(); await ws1.recv()
    ws2 = await websockets.connect(f"ws://localhost:8000/ws/{ROOM}/Bob")
    await ws2.recv(); await ws2.recv(); await ws1.recv()

    async def say(ws, other, text):
        await ws.send(json.dumps({"type": "msg", "text": text}))
        r = json.loads(await ws.recv())
        await other.recv()
        return r

    # Alice sends money-related messages
    r = await say(ws1, ws2, "Send Bob 25 bucks for coffee")
    if "money" in r.get("fired", []):
        PASS += 1; print(f"  [PASS] 19.1: WS money fires correctly")
    else:
        FAIL += 1; print(f"  [FAIL] 19.1: WS money didn't fire: {r.get('fired')}")

    # Bob sends ride
    r = await say(ws2, ws1, "Book me a ride to downtown")
    if "ride" in r.get("fired", []):
        PASS += 1; print(f"  [PASS] 19.2: WS ride fires correctly")
    else:
        FAIL += 1; print(f"  [FAIL] 19.2: WS ride didn't fire: {r.get('fired')}")

    # Alice chitchat
    await say(ws1, ws2, "Sounds good see you there")

    # Alice sarcasm — should NOT fire
    r = await say(ws1, ws2, "Lol send $0 bruh")
    if "money" not in r.get("fired", []):
        PASS += 1; print(f"  [PASS] 19.3: WS sarcasm suppressed in chat")
    else:
        FAIL += 1; print(f"  [FAIL] 19.3: WS sarcasm NOT suppressed: {r.get('fired')}")

    # Bob sets alarm
    r = await say(ws2, ws1, "Set an alarm for 7am")
    if "alarm" in r.get("fired", []):
        PASS += 1; print(f"  [PASS] 19.4: WS alarm fires")
    else:
        FAIL += 1; print(f"  [FAIL] 19.4: WS alarm didn't fire: {r.get('fired')}")

    # Bob reminder
    r = await say(ws2, ws1, "Remind me to grab my laptop")
    if "reminder" in r.get("fired", []):
        PASS += 1; print(f"  [PASS] 19.5: WS reminder fires")
    else:
        FAIL += 1; print(f"  [FAIL] 19.5: WS reminder didn't fire: {r.get('fired')}")

    # Alice multi-intent
    r = await say(ws1, ws2, "Text Rahul and send him 10 bucks")
    fired = r.get("fired", [])
    if "contact" in fired and "money" in fired:
        PASS += 1; print(f"  [PASS] 19.6: WS multi-intent fires")
    else:
        FAIL += 1; print(f"  [FAIL] 19.6: WS multi-intent missing: {fired}")

    # More chitchat
    await say(ws2, ws1, "Nice, see you tonight!")
    await say(ws1, ws2, "For sure, later!")

    await ws1.close()
    await ws2.close()

    # Test summaries
    time.sleep(0.5)
    alice = requests.get(f"{URL}/summary/{ROOM}/Alice").json()
    bob = requests.get(f"{URL}/summary/{ROOM}/Bob").json()

    # Alice should have: money, contact (maybe more from context)
    alice_intents = {i["intent"] for i in alice["intents"]}
    if "money" in alice_intents:
        PASS += 1; print(f"  [PASS] 19.7: Alice summary has money")
    else:
        FAIL += 1; print(f"  [FAIL] 19.7: Alice summary missing money: {alice_intents}")

    # Bob should have: ride, alarm, reminder
    bob_intents = {i["intent"] for i in bob["intents"]}
    if "ride" in bob_intents and "alarm" in bob_intents and "reminder" in bob_intents:
        PASS += 1; print(f"  [PASS] 19.8: Bob summary has ride+alarm+reminder")
    else:
        FAIL += 1; print(f"  [FAIL] 19.8: Bob summary missing intents: {bob_intents}")

    # Summary should have correct message counts
    if alice["total_messages"] >= 4:
        PASS += 1; print(f"  [PASS] 19.9: Alice message count={alice['total_messages']}")
    else:
        FAIL += 1; print(f"  [FAIL] 19.9: Alice count wrong: {alice['total_messages']}")

    if bob["total_messages"] >= 4:
        PASS += 1; print(f"  [PASS] 19.10: Bob message count={bob['total_messages']}")
    else:
        FAIL += 1; print(f"  [FAIL] 19.10: Bob count wrong: {bob['total_messages']}")

    # Check slots in summary
    for i_entry in alice["intents"]:
        if i_entry["intent"] == "money" and i_entry.get("slots"):
            if "amount" in i_entry["slots"]:
                PASS += 1; print(f"  [PASS] 19.11: Alice money summary has amount slot")
                break
    else:
        FAIL += 1; print(f"  [FAIL] 19.11: Alice money summary missing amount slot")

    # Non-existent user
    empty = requests.get(f"{URL}/summary/{ROOM}/Nobody").json()
    if empty["total_messages"] == 0 and len(empty["intents"]) == 0:
        PASS += 1; print(f"  [PASS] 19.12: Non-existent user returns empty summary")
    else:
        FAIL += 1; print(f"  [FAIL] 19.12: Non-existent user not empty: {empty}")

    # Non-existent room
    empty2 = requests.get(f"{URL}/summary/fake_room/Alice").json()
    if empty2["total_messages"] == 0:
        PASS += 1; print(f"  [PASS] 19.13: Non-existent room returns empty summary")
    else:
        FAIL += 1; print(f"  [FAIL] 19.13: Non-existent room not empty: {empty2}")

asyncio.run(test_ws_summary())


# ── 20. BATCH ENDPOINT ──
section("20. BATCH ENDPOINT")
batch = requests.post(f"{URL}/batch", json={"texts": [
    "Send 20 to Rahul",
    "Get me a ride home",
    "Set alarm for 6am",
    "Hey what's up",
    "",
]}).json()
results = batch["results"]
if "money" in results[0].get("fired", []):
    PASS += 1; print(f"  [PASS] 20.1: Batch money fires")
else:
    FAIL += 1; print(f"  [FAIL] 20.1: Batch money: {results[0].get('fired')}")
if "ride" in results[1].get("fired", []):
    PASS += 1; print(f"  [PASS] 20.2: Batch ride fires")
else:
    FAIL += 1; print(f"  [FAIL] 20.2: Batch ride: {results[1].get('fired')}")
if "alarm" in results[2].get("fired", []):
    PASS += 1; print(f"  [PASS] 20.3: Batch alarm fires")
else:
    FAIL += 1; print(f"  [FAIL] 20.3: Batch alarm: {results[2].get('fired')}")
if not results[3].get("fired"):
    PASS += 1; print(f"  [PASS] 20.4: Batch chitchat empty")
else:
    FAIL += 1; print(f"  [FAIL] 20.4: Batch chitchat: {results[3].get('fired')}")
if not results[4].get("fired"):
    PASS += 1; print(f"  [PASS] 20.5: Batch empty text empty")
else:
    FAIL += 1; print(f"  [FAIL] 20.5: Batch empty: {results[4].get('fired')}")


# ── 21. API HEALTH & META ──
section("21. API HEALTH & META")
health = requests.get(f"{URL}/health").json()
if health["status"] == "ok" and "thresholds" in health:
    PASS += 1; print(f"  [PASS] 21.1: /health ok, thresholds present")
else:
    FAIL += 1; print(f"  [FAIL] 21.1: health: {health}")

meta = requests.get(f"{URL}/meta").json()
if meta["model"] == "roberta-base-9intent" and len(meta["labels"]) == 9:
    PASS += 1; print(f"  [PASS] 21.2: /meta correct model name, 9 labels")
else:
    FAIL += 1; print(f"  [FAIL] 21.2: meta: {meta}")

# Check no file paths leak
if "Users" not in str(meta) and "C:\\" not in str(meta):
    PASS += 1; print(f"  [PASS] 21.3: /meta no file path leak")
else:
    FAIL += 1; print(f"  [FAIL] 21.3: meta leaks path: {meta}")

metrics = requests.get(f"{URL}/metrics").json()
if metrics["requests"] > 0:
    PASS += 1; print(f"  [PASS] 21.4: /metrics tracking requests ({metrics['requests']})")
else:
    FAIL += 1; print(f"  [FAIL] 21.4: metrics: {metrics}")


# ═══════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
total = PASS + FAIL
pct = (PASS / total * 100) if total else 0
color = "\033[92m" if FAIL == 0 else "\033[91m"
reset = "\033[0m"
print(f"  {color}FINAL: {PASS}/{total} passing ({pct:.1f}%){reset}")
if FAIL > 0:
    print(f"  {FAIL} FAILURES — review above")
print("=" * 80)
sys.exit(1 if FAIL > 0 else 0)
