"""
generate_synthetic.py
---------------------
Generates a 400-row synthetic student feedback CSV using Faker and pandas.
No personal identifiable information (PII) is included.

Usage:
    python data/generate_synthetic.py
"""

import random
import os
import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────────
CATEGORIES = [
    "academics", "hostel", "canteen", "sports",
    "library", "wifi", "administration", "mental_health",
]

DEPARTMENTS = ["CSE", "ECE", "ME", "CE", "MBA", "BBA", "Arts", "Science"]

SEVERITIES = ["low", "medium", "high"]

SEMESTERS = list(range(1, 9))

# ── Realistic feedback templates per category ──────────────────────────────────
TEMPLATES: dict[str, list[str]] = {
    "academics": [
        "The lecture pace in advanced algorithms is too fast; students fall behind by week three.",
        "Tutorial sessions are helpful but scheduled at inconvenient times.",
        "Course material for the data structures subject is outdated and needs revision.",
        "The professor explains concepts clearly and uses real-world examples effectively.",
        "Assignment deadlines cluster at the end of the semester, causing excessive stress.",
        "Elective course options are limited and do not cover emerging technologies.",
        "Lab sessions are well-organized and the equipment is adequate for the exercises.",
        "Mid-semester feedback from faculty is inconsistent across departments.",
        "The grading rubric for project submissions is unclear and subjective.",
        "Online resources shared by the faculty are very useful for self-study.",
        "Some theory classes lack practical demonstrations which affects understanding.",
        "The curriculum should include more industry-relevant case studies.",
        "Group project coordination is difficult due to lack of shared workspaces.",
        "Faculty office hours are rarely kept, making doubt resolution hard.",
        "The internal assessment weightage is disproportionate compared to end-semester exams.",
    ],
    "hostel": [
        "The hot water supply in Block C is unreliable during winter mornings.",
        "Room cleanliness inspections are infrequent and hygiene standards have dropped.",
        "The hostel common room TV is broken and has not been repaired for two months.",
        "Laundry facilities are insufficient for the number of residents in Block A.",
        "Night curfew rules are overly strict and do not accommodate late-night study sessions.",
        "The hostel warden is approachable and resolves complaints quickly.",
        "Power outages in the hostel are frequent and the backup generator is slow to start.",
        "Beds and mattresses in the new wing are comfortable and recently replaced.",
        "Security at the hostel gate is lax during late-night hours.",
        "The guest room facility is helpful for parents visiting during exam periods.",
        "Noise levels after 10 PM in the corridors disturb students who sleep early.",
        "Hostel fees were increased without adequate notice or justification.",
        "The recreation room equipment is old and some items are non-functional.",
        "Water pressure in bathrooms on upper floors is very low during peak hours.",
        "Pest control measures in the kitchen area are inadequate.",
    ],
    "canteen": [
        "The canteen food quality has declined significantly over the past semester.",
        "Vegetarian options are limited and often sold out by lunchtime.",
        "Prices for basic meals have increased without a corresponding quality improvement.",
        "The canteen is understaffed during peak hours, leading to long wait times.",
        "Hygiene standards in the food preparation area need urgent attention.",
        "The new menu introduced this semester offers better variety for different diets.",
        "Tea and coffee counters open too late, which is inconvenient for early-morning students.",
        "Tables are rarely wiped between uses, creating an unpleasant dining experience.",
        "The canteen timing does not align with the lecture schedule, causing students to skip meals.",
        "Regional cuisine options introduced recently are a welcome addition.",
        "Cold drinks and packed snacks are overpriced compared to nearby shops.",
        "The canteen app for pre-ordering food is a great initiative that saves time.",
        "Food served during evening hours is often left over from lunch and not fresh.",
        "Staff at the canteen are generally courteous and helpful.",
        "Special dietary needs such as gluten-free options are not catered for.",
    ],
    "sports": [
        "The football ground is in poor condition with uneven turf and waterlogging after rain.",
        "Sports equipment is not replaced regularly; some items are worn out.",
        "The badminton courts are well-maintained and available for booking online.",
        "Inter-department sports events should be organized more frequently.",
        "The gym lacks proper ventilation and becomes uncomfortably hot in summer.",
        "Coaching sessions for cricket are helpful and the coach is knowledgeable.",
        "Timings for sports facilities are too restrictive and do not suit academic schedules.",
        "The swimming pool was closed for maintenance for three months with no alternative arrangement.",
        "Sports achievements are celebrated in college events which motivates participation.",
        "There are no dedicated spaces for indoor games like table tennis or chess.",
        "Equipment booking system for the gym is disorganized and often causes conflicts.",
        "The annual sports meet is well-organized but lacks sufficient prizes and recognition.",
        "Floodlights on the basketball court need replacement for evening games.",
        "Yoga and fitness classes offered on weekends are popular and should be expanded.",
        "There is no provision for students with physical disabilities to participate in sports.",
    ],
    "library": [
        "The library's collection of reference books in mechanical engineering is outdated.",
        "Digital journal subscriptions are insufficient and many research papers are behind paywalls.",
        "The reading hall gets too crowded during exam time with no overflow space.",
        "Library staff are helpful in locating resources and suggesting alternatives.",
        "The online catalogue system is slow and often crashes during peak usage.",
        "Extended library hours during exam weeks are greatly appreciated.",
        "Book renewal via the mobile app is convenient but the app crashes frequently.",
        "The printing facility in the library is unreliable and expensive.",
        "New arrivals section is well-curated and updated regularly.",
        "Group study rooms in the library are in high demand and should be increased.",
        "Noise levels in certain sections of the library are distracting.",
        "The air conditioning in the library is too cold and makes it difficult to study for long.",
        "Inter-library loan facility would be very useful for specialized research.",
        "Wi-Fi connectivity inside the library is poor near the far end of the hall.",
        "Orientation sessions for new students on library resources are not provided.",
    ],
    "wifi": [
        "Wi-Fi connectivity in the hostel is extremely slow during evening hours.",
        "Dead zones near the lab building make accessing online resources during practicals impossible.",
        "The campus network blocks many educational websites without justification.",
        "Internet speed in the academic block has improved noticeably after the upgrade.",
        "Frequent disconnections during online exams are a serious concern for fairness.",
        "The VPN policy restricts access to certain research databases students need.",
        "Network authentication resets too frequently, requiring repeated logins.",
        "Wi-Fi coverage in the open-air amphitheatre is excellent.",
        "Bandwidth allocation per device is too low to support video lectures.",
        "Campus network downtime over the weekend is unacceptable for students with assignments due.",
        "The IT helpdesk response time for network issues exceeds 48 hours on average.",
        "Guest Wi-Fi for parents and visitors during events should be made available.",
        "Upload speeds are significantly lower than download speeds, affecting online submissions.",
        "The campus network firewall settings should be reviewed to allow legitimate academic tools.",
        "Streaming platforms are completely blocked, preventing use of educational video content.",
    ],
    "administration": [
        "Fee payment portal crashes frequently near the deadline, causing anxiety.",
        "Document verification for scholarship applications takes too long.",
        "The administration office has limited working hours that conflict with class schedules.",
        "Staff at the administrative counter are not always well-informed about procedures.",
        "Online application for bonafide certificates is a great time-saving initiative.",
        "Communication from the administration regarding schedule changes is often last-minute.",
        "Grievance redressal committee meetings are not held regularly.",
        "The scholarship disbursement process is opaque and lacks clear timelines.",
        "Exam hall allotment notices are released too close to the examination date.",
        "The new digital student portal streamlines document requests significantly.",
        "Re-evaluation applications are unnecessarily expensive and discourage students.",
        "Administration staff turnover is high, leading to inconsistent guidance.",
        "Emergency fund assistance for economically challenged students needs a simpler process.",
        "Timetable clashes are reported multiple times each semester without resolution.",
        "Parents have difficulty accessing progress reports online due to portal issues.",
    ],
    "mental_health": [
        "The counselling center wait times are too long; students in distress need faster access.",
        "Peer support programs are well-structured and have made a positive difference.",
        "Exam stress workshops offered in the wellness center are not widely publicized.",
        "There is a stigma around using mental health services that the college should actively address.",
        "The campus counsellor is empathetic and keeps sessions strictly confidential.",
        "A dedicated quiet room for students experiencing anxiety would be very beneficial.",
        "Mental health awareness week activities were engaging and informative.",
        "Faculty should receive training on recognizing signs of student distress.",
        "The workload in certain departments is unsustainable and contributes to burnout.",
        "Anonymous reporting system for mental health concerns would encourage more students to seek help.",
        "Study circles organized by the wellness team have improved student morale.",
        "Meditation and mindfulness sessions should be made part of the regular weekly schedule.",
        "Students returning after medical leave need better academic reintegration support.",
        "The college should increase the number of trained counsellors to reduce waiting times.",
        "Awareness materials about mental health resources are not visible enough across campus.",
    ],
}


def generate_feedback_text(category: str) -> str:
    """Return a randomly chosen realistic feedback sentence for the given category."""
    return random.choice(TEMPLATES[category])


def generate_dataset(n_rows: int = 400) -> pd.DataFrame:
    """
    Generate a synthetic student feedback dataset.

    Parameters
    ----------
    n_rows : int
        Number of feedback rows to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        feedback_id, department, semester, category, feedback_text, severity, date
    """
    records = []
    start_date = pd.Timestamp("2024-06-01")
    end_date = pd.Timestamp("2025-06-30")
    date_range_days = (end_date - start_date).days

    for i in range(1, n_rows + 1):
        category = random.choice(CATEGORIES)
        records.append(
            {
                "feedback_id": f"FB{i:04d}",
                "department": random.choice(DEPARTMENTS),
                "semester": random.choice(SEMESTERS),
                "category": category,
                "feedback_text": generate_feedback_text(category),
                "severity": random.choice(SEVERITIES),
                "date": (
                    start_date + pd.Timedelta(days=random.randint(0, date_range_days))
                ).strftime("%Y-%m-%d"),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    """Generate and save the synthetic feedback CSV."""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "synthetic_feedback.csv")

    print("Generating 400-row synthetic feedback dataset ...")
    df = generate_dataset(400)
    df.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")
    print(df.head(5).to_string(index=False))
    print(f"\nShape: {df.shape}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
