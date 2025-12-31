import database
from database import Database

# Initialize the database with the same websites
WEBSITES = [
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com",
    "https://prothomalo.com",
]
db = Database(WEBSITES)
db.init_database()  # Ensure tables exist (optional if already created)

# Open a session
session = db.Session()

# Query fingerprints
fingerprints = session.query(database.Fingerprint).all()
print("Fingerprints:")
for fp in fingerprints:
    print(f"ID: {fp.id}, Website: {fp.website}, Index: {fp.website_index}, Trace: {fp.trace_data[:50]}...")  # Truncate trace for readability

# Query collection stats
stats = session.query(database.CollectionStats).all()
print("\nCollection Stats:")
for stat in stats:
    print(f"Website: {stat.website}, Traces Collected: {stat.traces_collected}")

# Close the session
session.close()