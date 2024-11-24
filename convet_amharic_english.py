import csv

# File pairs and their splits
file_pairs = [
    ("train.am-en.base.am", "train.am-en.base.en", "train"),
    ("dev.am-en.base.am", "dev.am-en.base.en", "dev"),
    ("test.am-en.base.am", "test.am-en.base.en", "test")
]

output_file = "amharic_english_dataset.csv"
with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["language", "script", "sent_id", "text", "split"])

    # Iterate through file pairs
    for am_file, en_file, split in file_pairs:
        with open(am_file, "r", encoding="utf-8") as am, open(en_file, "r", encoding="utf-8") as en:
            am_lines = am.readlines()
            en_lines = en.readlines()

            assert len(am_lines) == len(en_lines), f"Mismatch in lines for {am_file} and {en_file}"

            for sent_id, (am_line, en_line) in enumerate(zip(am_lines, en_lines)):
                # Write Amharic row
                csvwriter.writerow(["amh", "Ethi", sent_id, am_line.strip(), split])
                # Write English row
                csvwriter.writerow(["eng", "Latn", sent_id, en_line.strip(), split])

print(f"Dataset saved to {output_file}")
