import os
from icrawler.builtin import BingImageCrawler

# Daftar kelas dan keyword pencarian
kelas_pohon = {
    "pohon_pisang": "pohon pisang"  # ← Tambahan baru
}

# Jumlah gambar per kelas
jumlah_gambar = 300

# Folder tujuan dataset
output_folder = os.path.join(os.getcwd(), "dataset")

# Membuat folder dataset jika belum ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Proses download untuk setiap kelas
for nama_folder, keyword in kelas_pohon.items():
    path_folder_kelas = os.path.join(output_folder, nama_folder)
    if not os.path.exists(path_folder_kelas):
        os.makedirs(path_folder_kelas)
    
    print(f"Mengunduh gambar untuk kelas: {nama_folder}...")

    crawler = BingImageCrawler(storage={"root_dir": path_folder_kelas})
    crawler.crawl(keyword=keyword, max_num=jumlah_gambar)

print("\n✅ Unduhan selesai! Gambar tersimpan dalam folder 'dataset'.")
