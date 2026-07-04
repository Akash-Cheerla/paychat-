"""
Real-world chat test for v20 — user-provided examples.
Tests natural everyday messages, sarcasm, typos, slang,
long passages, and full multi-turn conversations.
"""
import requests, json, random, time

URL = "http://localhost:8000"
PASS = 0
FAIL = 0
ALL = []

def send(text, room_id="default", sender="user1"):
    r = requests.post(f"{URL}/detect", json={
        "text": text, "chat_id": room_id, "sender": sender,
    }, timeout=30)
    return r.json()

def rid():
    return f"rc_{random.randint(10000,99999)}_{int(time.time()*1000)%100000}"

def t(label, text, expect, room_id=None, sender="user1"):
    """Test a single message. expect=[] means nothing should fire."""
    global PASS, FAIL
    if room_id is None:
        room_id = rid()
    r = send(text, room_id, sender)
    fired = set(r.get("intents", []))
    expected = set(expect)
    missing = expected - fired
    extra = fired - expected
    ok = not missing and not extra
    if ok:
        PASS += 1
        tag = "PASS"
    else:
        FAIL += 1
        tag = "FAIL"
    fired_s = ", ".join(sorted(fired)) if fired else "(none)"
    exp_s = ", ".join(sorted(expected)) if expected else "(none)"
    scores = {}
    for i in (expected | fired):
        if i in r.get("scores", {}):
            scores[i] = round(r["scores"][i], 4)
    trunc = text[:130] + "..." if len(text) > 130 else text
    print(f"  [{tag}] {trunc}")
    print(f"         expect={exp_s} | fired={fired_s}")
    if scores:
        print(f"         scores={scores}")
    if not ok:
        if missing: print(f"         !! MISSING: {missing}")
        if extra: print(f"         !! EXTRA: {extra}")
    ALL.append({"label": label, "text": text, "expected": list(expected),
                "fired": list(fired), "ok": ok, "scores": scores})
    return room_id

def section(name):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}\n")

# ══════════════════════════════════════════════════════════════
section("NORMAL EVERYDAY — should all be (none)")
# ══════════════════════════════════════════════════════════════
t("everyday_1", "I should probably head home soon.", [])
t("everyday_2", "Running a bit late, traffic is awful.", [])
t("everyday_3", "Can we push it to tomorrow?", [])
t("everyday_4", "I'll be there in like 20.", [])
t("everyday_5", "Today's been exhausting.", [])
t("everyday_6", "I'm starving.", [])
t("everyday_7", "I seriously need coffee.", [])
t("everyday_8", "Forgot my wallet again.", [])
t("everyday_9", "My phone is about to die.", [])
t("everyday_10", "I think I left my charger at your place.", [])

# ══════════════════════════════════════════════════════════════
section("LONGER MESSAGES — should all be (none)")
# ══════════════════════════════════════════════════════════════
t("longer_1", "I completely forgot I have a dentist appointment tomorrow morning. Good thing you reminded me yesterday or I would've missed it again.", [])
t("longer_2", "I don't think I'll be driving tonight. Parking downtown is impossible and I don't feel like dealing with traffic after work.", [])
t("longer_3", "Mom just texted saying the electricity bill is due this week. I swear these bills show up every five minutes.", [])
t("longer_4", "I'm meeting Alex after work but I haven't figured out how I'm getting there yet. Might just leave the car at home.", [])
t("longer_5", "I've been forgetting literally everything lately. My brain just isn't cooperating anymore.", [])

# ══════════════════════════════════════════════════════════════
section("SARCASM — should all be (none)")
# ══════════════════════════════════════════════════════════════
t("sarc_1", "Yeah because paying rent is my favorite hobby.", [])
t("sarc_2", "Sure, let me magically teleport there.", [])
t("sarc_3", "Guess my alarm clock decided to quit its job today.", [])
t("sarc_4", "Love spending my entire paycheck on bills.", [])
t("sarc_5", "Nothing screams adulthood like arguing with customer support.", [])
t("sarc_6", "Guess who's buying everyone dinner because they lost the bet...", [])

# ══════════════════════════════════════════════════════════════
section("HALF FINISHED THOUGHTS — should all be (none)")
# ══════════════════════════════════════════════════════════════
t("half_1", "Wait... actually never mind.", [])
t("half_2", "I was gonna... nah forget it.", [])
t("half_3", "You know what, screw it.", [])
t("half_4", "Hold on.", [])
t("half_5", "lemme think", [])

# ══════════════════════════════════════════════════════════════
section("MULTIPLE SMALL MESSAGES (combined) — should all be (none)")
# ══════════════════════════════════════════════════════════════
t("small_1", "Bro you awake need a favor", [])
t("small_2", "wait how much did i owe you again", [])
t("small_3", "okay forget it i figured it out", [])
t("small_4", "leave now or wait actually wait", [])

# ══════════════════════════════════════════════════════════════
section("WHATSAPP STYLE — should all be (none)")
# ══════════════════════════════════════════════════════════════
t("wa_1", "broooo where r u", [])
t("wa_2", "omw", [])
t("wa_3", "tmrw works better tbh", [])
t("wa_4", "idk man", [])
t("wa_5", "lmao", [])
t("wa_6", "bet", [])
t("wa_7", "fr?", [])
t("wa_8", "say less", [])

# ══════════════════════════════════════════════════════════════
section("WITH TYPOS — mixed")
# ══════════════════════════════════════════════════════════════
t("typo_1", "imma just ubre there", ["ride"])
t("typo_2", "remidn me later pls", ["reminder"])
t("typo_3", "payapl isnt working lol", [])
t("typo_4", "calender says im free", [])
t("typo_5", "lyfyt is cheaper rn", [])

# ══════════════════════════════════════════════════════════════
section("CONTEXT-BASED HARD — should all be (none)")
# ══════════════════════════════════════════════════════════════
t("ctx_hard_1", "I don't think I can make it if my car keeps acting up.", [])
t("ctx_hard_2", "It's almost midnight and I still haven't eaten anything.", [])
t("ctx_hard_3", "Tomorrow's gonna be packed.", [])
t("ctx_hard_4", "I seriously cannot miss this meeting.", [])
t("ctx_hard_5", "I'm gonna forget if I don't do something right now.", [])
t("ctx_hard_6", "I don't have enough cash on me.", [])
t("ctx_hard_7", "I hate carrying cash nowadays.", [])
t("ctx_hard_8", "Hopefully I don't oversleep again.", [])
t("ctx_hard_9", "Need to be at Pearson before six.", [])
t("ctx_hard_10", "The landlord already texted me twice.", [])

# ══════════════════════════════════════════════════════════════
section("MIXED CONTEXT — should all be (none)")
# ══════════════════════════════════════════════════════════════
t("mixed_1", "Let's grab dinner after work if you're free. Around 7 maybe? I don't feel like cooking today.", [])
t("mixed_2", "I'll probably leave my car here and figure something else out.", [])
t("mixed_3", "If you pay this time I'll get the next one.", [])
t("mixed_4", "Need to remember to call grandma before she gets mad.", [])
t("mixed_5", "I've got three meetings tomorrow and somehow they're all at the same time.", [])

# ══════════════════════════════════════════════════════════════
section("EMOTIONAL — should all be (none)")
# ══════════════════════════════════════════════════════════════
t("emo_1", "Today's just one of those days where nothing is going right.", [])
t("emo_2", "I'm honestly too mentally drained to think.", [])
t("emo_3", "Can today just be over already?", [])
t("emo_4", "Everything feels like it's piling up at once.", [])

# ══════════════════════════════════════════════════════════════
section("FUNNY — should all be (none)")
# ══════════════════════════════════════════════════════════════
t("funny_1", "My bank account just laughed at me.", [])
t("funny_2", "Adulting should come with cheat codes.", [])
t("funny_3", "The fridge has officially entered witness protection.", [])
t("funny_4", "My stomach is filing a complaint.", [])
t("funny_5", "Future me can deal with that problem.", [])

# ══════════════════════════════════════════════════════════════
section("PASSAGES (real chat style) — should all be (none)")
# ══════════════════════════════════════════════════════════════
t("passage_1", "I was planning to drive over after work, but honestly I don't think it's worth sitting in traffic for an hour just to find parking. If the weather stays like this I'll probably just figure something else out because I'm already exhausted.", [])
t("passage_2", "I completely forgot that tomorrow is Dad's birthday. I still haven't bought anything, and if I don't handle it tonight I'll definitely wake up tomorrow and panic at the last minute like I always do.", [])
t("passage_3", "I've spent the entire week trying to stay on top of everything, but somehow I still have rent due, my phone bill coming up, and about twenty other things I haven't even looked at yet. Being an adult is exhausting.", [])
t("passage_4", "Let's see how this evening goes. If everyone actually shows up on time, maybe we can grab dinner somewhere nearby. Worst case, we'll just order in because nobody's going to want to cook after today.", [])

# ══════════════════════════════════════════════════════════════
section("VERY HARD (no context, no intent) — should all be (none)")
# ══════════════════════════════════════════════════════════════
t("vhard_1", "Did you ever end up doing that?", [])
t("vhard_2", "Not yet.", [])
t("vhard_3", "Better before Friday.", [])
t("vhard_4", "You still good for tonight?", [])
t("vhard_5", "Depends.", [])
t("vhard_6", "Whether I can get there.", [])
t("vhard_7", "Did you remember?", [])
t("vhard_8", "Nope.", [])

# ══════════════════════════════════════════════════════════════
# MULTI-TURN CONVERSATIONS
# ══════════════════════════════════════════════════════════════

section("CHAT 1 — Friends Planning Dinner")
c1 = rid()
t("c1_01", "Yo, what are you up to tonight?", [], c1)
t("c1_02", "Nothing much, just got home.", [], c1)
t("c1_03", "Wanna grab dinner?", [], c1)
t("c1_04", "Sure, been craving burgers all day.", [], c1)
t("c1_05", "Around 7?", [], c1)
t("c1_06", "Yeah that works.", [], c1)
t("c1_07", "My car's still in the shop though.", [], c1)
t("c1_08", "Just Uber over.", ["ride"], c1)
t("c1_09", "Good idea.", ["ride"], c1)  # context boost
t("c1_10", "Where should we eat?", [], c1)
t("c1_11", "Doesn't matter honestly.", [], c1)
t("c1_12", "If it's packed we'll just order in.", [], c1)
t("c1_13", "Works for me.", [], c1)
t("c1_14", "I'll cover it.", [], c1)
t("c1_15", "Nah, I'll just send you my half afterwards.", ["money"], c1)

section("CHAT 2 — Forgetful Person")
c2 = rid()
t("c2_01", "I swear my memory is getting worse.", [], c2)
t("c2_02", "Why?", [], c2)
t("c2_03", "Forgot my laptop at work yesterday.", [], c2)
t("c2_04", "Classic.", [], c2)
t("c2_05", "And tomorrow I need to submit those documents.", [], c2)
t("c2_06", "Don't forget.", [], c2)
t("c2_07", "I probably will.", [], c2)
t("c2_08", "Then remind yourself tonight before bed.", [], c2)
t("c2_09", "Yeah that's probably the safest option.", [], c2)

section("CHAT 3 — Airport (alarm + ride expected)")
c3 = rid()
t("c3_01", "My flight's ridiculously early tomorrow.", [], c3)
t("c3_02", "What time?", [], c3)
t("c3_03", "6:10 AM.", [], c3)
t("c3_04", "Ouch.", [], c3)
t("c3_05", "Which means I have to leave before 4.", [], c3)
t("c3_06", "That's brutal.", [], c3)
t("c3_07", "I definitely can't oversleep again.", [], c3)
t("c3_08", "Nope.", [], c3)
t("c3_09", "I'll probably just Uber to the airport.", ["ride"], c3)
t("c3_10", "Easier than paying airport parking.", [], c3)

section("CHAT 4 — Bills")
c4 = rid()
t("c4_01", "My landlord texted me again.", [], c4)
t("c4_02", "About rent?", [], c4)
t("c4_03", "Yep.", [], c4)
t("c4_04", "Already?", [], c4)
t("c4_05", "Feels like I just paid it.", [], c4)
t("c4_06", "Welcome to adulthood.", [], c4)
t("c4_07", "Need to make sure I don't forget this month.", [], c4)
t("c4_08", "Better handle it before Friday.", [], c4)

section("CHAT 5 — Just Friends (NO INTENT)")
c5 = rid()
t("c5_01", "Have you watched that new show yet?", [], c5)
t("c5_02", "Not yet.", [], c5)
t("c5_03", "It's actually really good.", [], c5)
t("c5_04", "Everyone keeps saying that.", [], c5)
t("c5_05", "You'd probably binge it in one night.", [], c5)
t("c5_06", "Sounds about right.", [], c5)
t("c5_07", "I'll start it this weekend.", [], c5)

section("CHAT 6 — Indirect Ride")
c6 = rid()
t("c6_01", "This rain is insane.", [], c6)
t("c6_02", "Tell me about it.", [], c6)
t("c6_03", "Walking home doesn't sound fun anymore.", [], c6)
t("c6_04", "Definitely not.", [], c6)
t("c6_05", "Guess I'll figure something out.", [], c6)

section("CHAT 7 — Payment Without Saying Pay")
c7 = rid()
t("c7_01", "Thanks again for covering lunch.", [], c7)
t("c7_02", "No worries.", [], c7)
t("c7_03", "I still owe you.", ["money"], c7)
t("c7_04", "Yeah", ["money"], c7)  # context
t("c7_05", "I'll get it back to you tonight.", ["money"], c7)

section("CHAT 8 — Food Order Indirect")
c8 = rid()
t("c8_01", "What's for dinner?", [], c8)
t("c8_02", "Absolutely nothing.", [], c8)
t("c8_03", "Too lazy to cook.", [], c8)
t("c8_04", "Same.", [], c8)
t("c8_05", "Guess we're ordering something.", ["food_order"], c8)

section("CHAT 9 — Context Only (calendar-ish)")
c9 = rid()
t("c9_01", "Are we still on?", [], c9)
t("c9_02", "Yep.", [], c9)
t("c9_03", "Same place?", [], c9)
t("c9_04", "Yeah.", [], c9)
t("c9_05", "See you around 6 then.", [], c9)

section("CHAT 10 — Contacts")
c10 = rid()
t("c10_01", "Do you still have Ethan's number?", [], c10)
t("c10_02", "Yeah.", [], c10)
t("c10_03", "Can you send it?", ["contact"], c10)
t("c10_04", "Sure.", ["contact"], c10)  # context

section("CHAT 11 — Sarcasm about Bills")
c11 = rid()
t("c11_01", "My favorite time of the month.", [], c11)
t("c11_02", "Payday?", [], c11)
t("c11_03", "Nope.", [], c11)
t("c11_04", "Bills.", [], c11)
t("c11_05", "Love giving everyone my money.", [], c11)

section("CHAT 12 — Multi Intent")
c12 = rid()
t("c12_01", "Tomorrow's dinner is still happening right?", [], c12)
t("c12_02", "Yep.", [], c12)
t("c12_03", "Around 8?", [], c12)
t("c12_04", "Yeah.", [], c12)
t("c12_05", "Remind me tomorrow afternoon.", ["reminder"], c12)
t("c12_06", "Will do.", ["reminder"], c12)  # context
t("c12_07", "I don't think I'm driving.", [], c12)
t("c12_08", "Uber?", [], c12)
t("c12_09", "Probably.", [], c12)
t("c12_10", "I'll pay you back for the tickets too.", ["money"], c12)

section("CHAT 13 — Relationship Chat (NO INTENT)")
c13 = rid()
t("c13_01", "Are you mad?", [], c13)
t("c13_02", "No.", [], c13)
t("c13_03", "You seem different.", [], c13)
t("c13_04", "Just tired.", [], c13)
t("c13_05", "You sure?", [], c13)
t("c13_06", "Yeah.", [], c13)
t("c13_07", "Okay.", [], c13)

section("CHAT 14 — Busy Day")
c14 = rid()
t("c14_01", "Tomorrow is going to be chaos.", [], c14)
t("c14_02", "Why?", [], c14)
t("c14_03", "Gym. Dentist. Team meeting. Dinner with family.", [], c14)
t("c14_04", "Busy day.", [], c14)
t("c14_05", "If I don't write this down I'll forget something.", [], c14)

section("CHAT 15 — Parents + Bills")
c15 = rid()
t("c15_01", "Did you pay the internet bill?", [], c15)
t("c15_02", "Not yet.", [], c15)
t("c15_03", "It expires tomorrow.", [], c15)
t("c15_04", "I'll take care of it tonight.", ["bills"], c15)
t("c15_05", "Thanks.", [], c15)

section("CHAT 16 — Weekend Planning")
c16 = rid()
t("c16_01", "You doing anything Saturday?", [], c16)
t("c16_02", "Not really.", [], c16)
t("c16_03", "Want to go hiking?", [], c16)
t("c16_04", "Sounds fun.", [], c16)
t("c16_05", "Morning?", [], c16)
t("c16_06", "Sure.", [], c16)

section("CHAT 17 — False Positive: Bill Burr (NO INTENT)")
c17 = rid()
t("c17_01", "Bill Burr's new special is hilarious.", [], c17)
t("c17_02", "I watched it yesterday.", [], c17)
t("c17_03", "Dude is so funny.", [], c17)

section("CHAT 18 — False Positive: roller coaster ride (NO INTENT)")
c18 = rid()
t("c18_01", "That roller coaster ride was insane.", [], c18)
t("c18_02", "My heart almost stopped.", [], c18)
t("c18_03", "Worth it though.", [], c18)
t("c18_04", "Absolutely.", [], c18)

section("CHAT 19 — Very Real WhatsApp")
c19 = rid()
t("c19_01", "bro", [], c19)
t("c19_02", "yo", [], c19)
t("c19_03", "u free later", [], c19)
t("c19_04", "maybe why", [], c19)
t("c19_05", "was thinking food", [], c19)
t("c19_06", "down", [], c19)
t("c19_07", "cool", [], c19)
t("c19_08", "where", [], c19)
t("c19_09", "idk", [], c19)
t("c19_10", "we'll figure it out", [], c19)
t("c19_11", "sounds good", [], c19)
t("c19_12", "if parking sucks im not driving lol", [], c19)
t("c19_13", "fair", [], c19)
t("c19_14", "worst case we'll order", ["food_order"], c19)
t("c19_15", "bet", [], c19)

section("CHAT 20 — Long Mixed Conversation")
c20 = rid()
t("c20_01", "I completely forgot tomorrow's Friday already.", [], c20)
t("c20_02", "This week flew by.", [], c20)
t("c20_03", "We still have that client meeting in the morning.", [], c20)
t("c20_04", "Yep, 9 AM.", [], c20)
t("c20_05", "I seriously can't oversleep tomorrow.", [], c20)
t("c20_06", "Me neither.", [], c20)
t("c20_07", "After work though, we should celebrate.", [], c20)
t("c20_08", "Definitely.", [], c20)
t("c20_09", "Let's grab sushi.", [], c20)
t("c20_10", "I'm in.", [], c20)
t("c20_11", "Around 7?", [], c20)
t("c20_12", "Works.", [], c20)
t("c20_13", "If traffic is bad I'm leaving my car at home.", [], c20)
t("c20_14", "Probably easier.", [], c20)
t("c20_15", "Yeah, I'll just get there another way.", [], c20)
t("c20_16", "Good call.", [], c20)
t("c20_17", "Also remind me to bring those documents.", ["reminder"], c20)
t("c20_18", "Got it.", ["reminder"], c20)  # context
t("c20_19", "I'll cover dinner.", [], c20)
t("c20_20", "Nah we'll split it.", [], c20)

# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════
total = PASS + FAIL
pct = 100 * PASS / total if total else 0
print(f"\n{'='*70}")
print(f"  REAL CHAT TEST: {PASS}/{total} ({pct:.1f}%)")
print(f"  Passed: {PASS} | Failed: {FAIL}")
print(f"{'='*70}")

# Failures summary
fails = [r for r in ALL if not r["ok"]]
if fails:
    print(f"\n  FAILURES ({len(fails)}):")
    for f in fails:
        exp_s = ", ".join(f["expected"]) if f["expected"] else "(none)"
        fir_s = ", ".join(f["fired"]) if f["fired"] else "(none)"
        trunc = f["text"][:100] + "..." if len(f["text"]) > 100 else f["text"]
        print(f"    [{f['label']}] \"{trunc}\"")
        print(f"      expected={exp_s} fired={fir_s} {f['scores']}")

with open("realchat_v20_results.json", "w") as fout:
    json.dump({"pass": PASS, "fail": FAIL, "total": total, "results": ALL}, fout, indent=2)
print(f"\nSaved to realchat_v20_results.json")
