#!/usr/bin/env python3
# generate_100_test_emails.py
# Creates 100 synthetic test emails (50 ham, 50 spam) in ./test/
# Also writes ./test/test_labels.csv with ground-truth labels (0=ham, +1=spam).
import os, random, datetime
random.seed(42)

OUT_DIR = "src/data/test"
os.makedirs(OUT_DIR, exist_ok=True)

# templates
ham_templates = [
    "Hi {name},\n\nJust a reminder about the {event} on {date}. Please confirm your attendance.\n\nThanks,\n{sender}",
    "Hello {name},\n\nAttached are the notes from today's meeting. Let me know if anything is missing.\n\nRegards,\n{sender}",
    "Hey {name},\n\nQuick question: are you available for a short call tomorrow afternoon?\n\nCheers,\n{sender}",
    "Dear {name},\n\nI've updated the project roadmap and pushed changes to the repository. Please review when free.\n\nBest,\n{sender}",
    "Hi {name},\n\nThank you for your feedback on the draft. I incorporated most suggestions and uploaded version 2.\n\nSincerely,\n{sender}"
]

spam_templates = [
    "CONGRATULATIONS {name}! You have been selected to WIN a {prize}. Click here to claim: {url}",
    "URGENT: Your account is compromised. Verify now at {url} to avoid suspension.",
    "Earn ${amount}/week working from home! No experience required. Sign up: {url}",
    "Limited time offer: Buy one get one free on {product}. Visit {url} and use code {code}.",
    "You won a special reward! Provide bank details at {url} to receive ${amount}."
]

names = ["Alex","Sam","Jordan","Taylor","Riley","Morgan","Casey","Jamie","Avery","Cameron"]
senders = ["Priya","Akash","John","Anita","Suresh","Meera","Dev","Lina","Rahul","Nina"]
events = ["project demo","team meeting","client call","presentation","seminar"]
products = ["smartphone","headphones","laptop","gift card","subscription"]
codes = ["WIN2025","FREE50","OFFER99","CLAIMIT","GETNOW"]
domains = ["example.com","promo-site.net","fastpay.io","reward-center.org","secure-login.info"]

def make_ham(i):
    tmpl = random.choice(ham_templates)
    name = random.choice(names)
    sender = random.choice(senders)
    event = random.choice(events)
    date = (datetime.date.today() + datetime.timedelta(days=random.randint(1,14))).isoformat()
    return tmpl.format(name=name, event=event, date=date, sender=sender)

def make_spam(i):
    tmpl = random.choice(spam_templates)
    name = random.choice(names)
    prize = random.choice(products).title()
    amount = random.randint(50,5000)
    url = f"https://{random.choice(domains)}/claim/{random.randint(10000,99999)}"
    product = random.choice(products)
    code = random.choice(codes)
    return tmpl.format(name=name, prize=prize, amount=amount, url=url, product=product, code=code)

# generate 100 emails: alternating ham/spam roughly to mix order
labels = []
idx = 1
# produce 50 ham + 50 spam, shuffled order
pool = []
for _ in range(50):
    pool.append(("ham", make_ham(idx)))
    idx += 1
for _ in range(50):
    pool.append(("spam", make_spam(idx)))
    idx += 1
random.shuffle(pool)

# write files
for i,(lab,text) in enumerate(pool, start=1):
    fname = f"email{i}.txt"
    with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
        f.write(text)
    labels.append((fname, "+1" if lab=="spam" else "0"))

# write labels CSV
import csv
with open(os.path.join(OUT_DIR, "test_labels.csv"), "w", newline='', encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["filename","label"])  # label: +1 spam, 0 ham
    writer.writerows(labels)

print(f"Created {len(pool)} test emails in '{OUT_DIR}/' and '{OUT_DIR}/test_labels.csv'.")
