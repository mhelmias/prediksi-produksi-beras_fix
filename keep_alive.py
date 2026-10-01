import os
import sys

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

URL = os.environ.get("APP_URL", "https://estimasi-beras.streamlit.app/")

# Label ini hanya muncul setelah app berhasil mengambil data dari Supabase,
# jadi sekaligus membuktikan koneksi database masih hidup.
TEKS_SUKSES = "Total estimasi beras"
TEKS_ERROR = "Aplikasi mengalami kesalahan"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=60_000)

    # Jika app tertidur, klik tombol wake up
    wake = page.get_by_role("button", name="Yes, get this app back up!")
    try:
        wake.wait_for(timeout=10_000)
        wake.click()
        print("App tertidur -> tombol wake up diklik")
    except PWTimeout:
        print("App sudah aktif, tidak perlu wake up")

    # Streamlit Cloud menampilkan app di dalam iframe
    frame = page.frame_locator('iframe[title="streamlitApp"]')
    sukses = frame.get_by_text(TEKS_SUKSES).first
    error = frame.get_by_text(TEKS_ERROR).first

    try:
        sukses.or_(error).wait_for(timeout=180_000)
    except PWTimeout:
        print("GAGAL: dashboard tidak termuat dalam 3 menit")
        browser.close()
        sys.exit(1)

    if error.is_visible():
        print("GAGAL: app aktif tetapi menampilkan pesan error (cek Supabase)")
        browser.close()
        sys.exit(1)

    # Biarkan koneksi terbuka sebentar agar terhitung sebagai kunjungan
    page.wait_for_timeout(20_000)
    print("OK: dashboard aktif dan data berhasil dimuat")
    browser.close()
