import csv

def increment_first_column(csv_file_path):
    # Read the CSV file
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Iterate through each row and increment the first element by 1
    for row in rows:
        if row and row[0].isdigit():  # Check if first element is a number
            row[0] = str(int(row[0]) + 1)

    # Write the updated rows back to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

# Example usage
csv_file_path = 'testing.csv'
increment_first_column(csv_file_path)