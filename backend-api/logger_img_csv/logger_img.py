import csv
import os

# Ghi log vào CSV cho phân tích ảnh
def log_image_analysis(image_path, image_description):
    log_data = [
        {'Image Path': image_path, 
         'Image Description': image_description}
    ]
    
    # Ghi log vào file CSV (append mode)
    with open('image_analysis_log.csv', mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=log_data[0].keys())
        if os.path.getsize('image_analysis_log.csv') == 0:  # Nếu file CSV rỗng, ghi tiêu đề cột
            writer.writeheader()
        writer.writerows(log_data)
    print(f"Log đã được ghi vào image_analysis_log.csv")