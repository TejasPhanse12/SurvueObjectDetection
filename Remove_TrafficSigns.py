import os

def remove_trafficsign(labels_dir):
    for filename in os.listdir(labels_dir):
        if not filename.endswith('.txt'):
            continue
        
        filepath = os.path.join(labels_dir, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Keep 0 (Dummy), 1 (human), 3 (vehicle)
        # Remap: 0 → 0, 1 → 1, 3 → 2
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if class_id == 0:  # Dummy → 0
                    new_lines.append(line)
                elif class_id == 1:  # human → 1
                    new_lines.append(line)
                elif class_id == 3:  # vehicle → 2
                    parts[0] = '2'
                    new_lines.append(' '.join(parts) + '\n')
                # Skip class 2 (trafficsign)
        
        with open(filepath, 'w') as f:
            f.writelines(new_lines)

# Run on both train and val
remove_trafficsign('dataset_copy/train/labels')
remove_trafficsign('dataset_copy/val/labels')
print("Done! Traffic signs removed.")
 