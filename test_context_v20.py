"""
Comprehensive context chain tests for v20.
Tests multi-message conversations with context carry-forward.
Each test uses a unique room_id to avoid cross-contamination.
"""
import requests, json, time, random

URL = "http://localhost:8000"
PASS = 0
FAIL = 0

def send(text, room_id, sender="user1"):
    r = requests.post(f"{URL}/detect", json={
        "text": text,
        "chat_id": room_id,
        "sender": sender,
    }, timeout=30)
    return r.json()

def rid():
    return f"ctx_{random.randint(10000,99999)}_{int(time.time()*1000)%100000}"

def check_chain(chain_name, messages, room_id=None):
    """Run a chain of messages and check each one."""
    global PASS, FAIL
    if room_id is None:
        room_id = rid()

    print(f"\n{'='*70}")
    print(f"  CHAIN: {chain_name}")
    print(f"  room: {room_id}")
    print(f"{'='*70}")

    for i, (text, expect_intents, expect_not) in enumerate(messages):
        r = send(text, room_id)
        fired = set(r.get("intents", []))
        expect = set(expect_intents) if expect_intents else set()
        expect_not_set = set(expect_not) if expect_not else set()

        missing = expect - fired
        unwanted = fired & expect_not_set
        ok = not missing and not unwanted

        # For no-intent expected, any fire is a fail
        if not expect_intents and fired:
            # Check if any fired intent is in expect_not
            if expect_not and unwanted:
                ok = False
            elif not expect_not:
                # If we expected nothing and something fired, that's wrong
                ok = False

        if ok:
            PASS += 1
            status = "PASS"
        else:
            FAIL += 1
            status = "FAIL"

        fired_str = ", ".join(sorted(fired)) if fired else "(none)"
        expect_str = ", ".join(sorted(expect)) if expect else "(none)"

        scores_relevant = {}
        for intent in (expect | fired):
            if intent in r.get("scores", {}):
                scores_relevant[intent] = round(r["scores"][intent], 4)

        action = r.get("action_score")
        action_str = f" action={action:.3f}" if action else ""
        boosted = r.get("context_boosted")
        boost_str = f" boosted={boosted}" if boosted else ""

        print(f"  [{status}] msg{i+1}: \"{text}\"")
        print(f"         expected={expect_str} | fired={fired_str}{action_str}{boost_str}")
        if scores_relevant:
            print(f"         scores={scores_relevant}")
        if not ok:
            if missing:
                print(f"         !! MISSING: {missing}")
            if unwanted:
                print(f"         !! UNWANTED: {unwanted}")


# ══════════════════════════════════════════════════════════════
#  1. FOOD CONTEXT CHAINS
# ══════════════════════════════════════════════════════════════

check_chain("Food: direct order then follow-up", [
    ("Order me a pizza", ["food_order"], []),
    ("Make it a large pepperoni", ["food_order"], []),
    ("Add garlic bread too", ["food_order"], []),
])

check_chain("Food: discussion then order", [
    ("I'm so hungry right now", [], []),
    ("What should we eat?", [], []),
    ("Just order pizza already", ["food_order"], []),
])

check_chain("Food: order then confirmation", [
    ("Get me some chinese food", ["food_order"], []),
    ("Sounds good", ["food_order"], []),
    ("Yeah do it", ["food_order"], []),
])

check_chain("Food: order then cancel", [
    ("Order some tacos from that place", ["food_order"], []),
    ("Actually nah never mind", [], []),
])

check_chain("Food: order then unrelated chat", [
    ("Get me a burger from mcdonalds", ["food_order"], []),
    ("Did you see the game last night?", [], ["food_order"]),
    ("That play was insane", [], ["food_order"]),
])

check_chain("Food: vague then specific", [
    ("I could eat something", [], []),
    ("Order me some sushi", ["food_order"], []),
])

# ══════════════════════════════════════════════════════════════
#  2. RIDE CONTEXT CHAINS
# ══════════════════════════════════════════════════════════════

check_chain("Ride: direct request then follow-up", [
    ("Book me an uber", ["ride"], []),
    ("To the airport", ["ride"], []),
    ("Make it an uber XL", ["ride"], []),
])

check_chain("Ride: request then confirmation", [
    ("Get me a ride to downtown", ["ride"], []),
    ("Yes do it", ["ride"], []),
])

check_chain("Ride: request then cancel", [
    ("Call me a cab to the station", ["ride"], []),
    ("Wait cancel that actually", [], []),
])

check_chain("Ride: discussion then request", [
    ("Traffic is terrible right now", [], []),
    ("Uber prices are crazy", [], []),
    ("Book me one anyway", ["ride"], []),
])

check_chain("Ride: request then unrelated", [
    ("Pick me up from the mall", ["ride"], []),
    ("Bro have you watched that new show", [], ["ride"]),
])

# ══════════════════════════════════════════════════════════════
#  3. MONEY CONTEXT CHAINS
# ══════════════════════════════════════════════════════════════

check_chain("Money: request then confirmation", [
    ("Send Rahul 50 bucks", ["money"], []),
    ("Yeah go ahead", ["money"], []),
])

check_chain("Money: request then follow-up details", [
    ("Venmo me for lunch", ["money"], []),
    ("It was 25 dollars", ["money"], []),
])

check_chain("Money: multi-step payment", [
    ("I owe you from last week", ["money"], []),
    ("How much was it again", ["money"], []),
    ("Send it", ["money"], []),
])

check_chain("Money: request then cancel", [
    ("Pay Jake 30 for the tickets", ["money"], []),
    ("Nah wait dont do that", [], []),
])

check_chain("Money: request then unrelated", [
    ("Zelle me 20 bucks", ["money"], []),
    ("Hey what time is the party", [], ["money"]),
])

check_chain("Money: discussion not request", [
    ("I spent way too much money this week", [], []),
    ("Same bro my wallet is crying", [], []),
    ("We really need to budget better", [], []),
])

# ══════════════════════════════════════════════════════════════
#  4. ALARM CONTEXT CHAINS
# ══════════════════════════════════════════════════════════════

check_chain("Alarm: set then modify", [
    ("Set an alarm for 7am", ["alarm"], []),
    ("Actually make it 6:30", ["alarm"], []),
])

check_chain("Alarm: set then confirm", [
    ("Wake me up at 8 tomorrow", ["alarm"], []),
    ("Yeah do that", ["alarm"], []),
])

check_chain("Alarm: multi-alarm chain", [
    ("Set alarm for 6am", ["alarm"], []),
    ("And one for 6:30 too", ["alarm"], []),
    ("And a backup at 7", ["alarm"], []),
])

# ══════════════════════════════════════════════════════════════
#  5. REMINDER CONTEXT CHAINS
# ══════════════════════════════════════════════════════════════

check_chain("Reminder: set then confirm", [
    ("Remind me to call mom tomorrow", ["reminder"], []),
    ("Yeah please", ["reminder"], []),
])

check_chain("Reminder: set then add detail", [
    ("Remind me to buy groceries", ["reminder"], []),
    ("From trader joes specifically", ["reminder"], []),
])

check_chain("Reminder: set then cancel", [
    ("Set a reminder for the dentist appointment", ["reminder"], []),
    ("Never mind I already rescheduled it", [], []),
])

# ══════════════════════════════════════════════════════════════
#  6. CONTACT CONTEXT CHAINS
# ══════════════════════════════════════════════════════════════

check_chain("Contact: call then confirm", [
    ("Call Rahul and tell him were on the way", ["contact"], []),
    ("Yeah go ahead", ["contact"], []),
])

check_chain("Contact: save then another", [
    ("Save this number as Pizza Place", ["contact"], []),
    ("And save this one as Gym Reception", ["contact"], []),
])

check_chain("Contact: text then unrelated", [
    ("Text Mom that I'll be late", ["contact"], []),
    ("What are we having for dinner", [], ["contact"]),
])

# ══════════════════════════════════════════════════════════════
#  7. CALENDAR CONTEXT CHAINS
# ══════════════════════════════════════════════════════════════

check_chain("Calendar: schedule then confirm", [
    ("Block 2pm to 4pm for the meeting", ["calendar"], []),
    ("Yes confirm that", ["calendar"], []),
])

check_chain("Calendar: schedule then modify", [
    ("Put dentist on thursday", ["calendar"], []),
    ("Actually move it to friday", ["calendar"], []),
])

# ══════════════════════════════════════════════════════════════
#  8. CROSS-INTENT CONTEXT (intent switch mid-conversation)
# ══════════════════════════════════════════════════════════════

check_chain("Cross: food then money", [
    ("Order me a pizza", ["food_order"], []),
    ("And venmo jake for his share", ["money"], []),
])

check_chain("Cross: ride then contact", [
    ("Book me an uber to the airport", ["ride"], []),
    ("And text rahul that im coming", ["contact"], []),
])

check_chain("Cross: money then alarm", [
    ("Send Sarah 20 bucks", ["money"], []),
    ("And set an alarm for 6am", ["alarm"], []),
])

check_chain("Cross: food then ride then chitchat", [
    ("Order some pizza for later", ["food_order"], []),
    ("And get me a ride home first", ["ride"], []),
    ("This day has been crazy lol", [], []),
])

check_chain("Cross: multiple intents then generic confirm", [
    ("Venmo jake 20 bucks", ["money"], []),
    ("And remind me to call him tomorrow", ["reminder"], []),
    ("Do both", ["money", "reminder"], []),  # context boost both
])

# ══════════════════════════════════════════════════════════════
#  9. AMBIGUOUS FOLLOW-UPS (context should help disambiguate)
# ══════════════════════════════════════════════════════════════

check_chain("Ambiguous: 'do it' after ride", [
    ("Get me an uber to downtown", ["ride"], []),
    ("Do it", ["ride"], []),
])

check_chain("Ambiguous: 'do it' after money", [
    ("Venmo me 30 bucks", ["money"], []),
    ("Do it", ["money"], []),
])

check_chain("Ambiguous: 'do it' after food", [
    ("Order me chinese food", ["food_order"], []),
    ("Do it", ["food_order"], []),
])

check_chain("Ambiguous: 'sounds good' after alarm", [
    ("Set alarm for 7am", ["alarm"], []),
    ("Sounds good", ["alarm"], []),
])

check_chain("Ambiguous: 'yes' after reminder", [
    ("Remind me to take medicine at 8", ["reminder"], []),
    ("Yes", ["reminder"], []),
])

check_chain("Ambiguous: 'lets go' after ride", [
    ("Book a lyft to the station", ["ride"], []),
    ("Let's go", ["ride"], []),
])

check_chain("Ambiguous: 'send it' after money", [
    ("I owe jake 50 for the tickets", ["money"], []),
    ("Send it", ["money"], []),
])

# ══════════════════════════════════════════════════════════════
#  10. CONTEXT SHOULD NOT CARRY (different room / too far apart)
# ══════════════════════════════════════════════════════════════

room_a = rid()
room_b = rid()
check_chain("Isolation: intent in room A, chitchat in room B", [
    ("Order me a pizza", ["food_order"], []),
], room_id=room_a)

check_chain("Isolation: room B should not see room A context", [
    ("Sounds good", [], []),
], room_id=room_b)

check_chain("No context: generic 'do it' with no prior intent", [
    ("Do it", [], []),
])

check_chain("No context: 'yes' with no prior intent", [
    ("Yes", [], []),
])

check_chain("No context: 'sounds good' standalone", [
    ("Sounds good", [], []),
])

# ══════════════════════════════════════════════════════════════
#  11. LONG CHAINS (3+ messages of context)
# ══════════════════════════════════════════════════════════════

check_chain("Long: 5-message food chain", [
    ("What should we eat", [], []),
    ("I'm thinking pizza", [], []),
    ("Or maybe chinese", [], []),
    ("Just order pizza bro", ["food_order"], []),
    ("Large pepperoni", ["food_order"], []),
])

check_chain("Long: 4-message money chain", [
    ("Hey you owe me from last week", ["money"], []),
    ("It was 30 dollars remember", ["money"], []),
    ("Venmo me", ["money"], []),
    ("Right now please", ["money"], []),
])

check_chain("Long: mixed chain with topic shifts", [
    ("Book me an uber", ["ride"], []),
    ("Actually how much is it", [], []),
    ("Nah thats too expensive", [], []),
    ("I'll just take the bus", [], ["ride"]),
])

# ══════════════════════════════════════════════════════════════
#  12. EDGE CASES
# ══════════════════════════════════════════════════════════════

check_chain("Edge: same intent repeated", [
    ("Venmo me 20", ["money"], []),
    ("Venmo me 30", ["money"], []),
    ("Venmo me 50", ["money"], []),
])

check_chain("Edge: sarcasm after real intent", [
    ("Send Jake 20 bucks", ["money"], []),
    ("Yeah right and a million dollars lmao", [], ["money"]),
])

check_chain("Edge: question after intent", [
    ("Order me sushi", ["food_order"], []),
    ("Wait is that place even open?", [], []),
])

check_chain("Edge: emoji/short response after intent", [
    ("Book me an uber to the mall", ["ride"], []),
    ("bet", [], []),
])

check_chain("Edge: typo-like message after intent", [
    ("Set alarm for 6am", ["alarm"], []),
    ("ok", ["alarm"], []),
])

# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════

total = PASS + FAIL
print(f"\n{'='*70}")
print(f"  CONTEXT TEST RESULTS: {PASS}/{total} passing ({100*PASS/total:.1f}%)")
print(f"  Failures: {FAIL}")
print(f"{'='*70}")
