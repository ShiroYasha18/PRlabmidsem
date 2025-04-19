import re

# Define regex patterns
patterns = {
    "Email": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
    "Phone Number": r"^\d{10}$",  # Matches a 10-digit number
    "Date (YYYY-MM-DD)": r"^\d{4}-\d{2}-\d{2}$"
}

# Get user input
user_input = input("Enter something: ")

# Check for matches
for name, pattern in patterns.items():
    if re.match(pattern, user_input):
        print(f"Matched: {name}")
        break
else:
    print("No match found.")
