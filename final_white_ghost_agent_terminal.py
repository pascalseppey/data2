#!/usr/bin/env python3

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
import zipfile
import random
import time
import os
import tempfile
import signal
import subprocess

def kill_existing_chrome_sessions():
    print("🧹 Nettoyage des anciennes sessions Chrome...")

    try:
        # Trouve les PID des processus chrome/chromedriver
        chrome_pids = subprocess.check_output("pgrep -f '(chrome|chromedriver)'", shell=True).decode().splitlines()

        for pid in chrome_pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
                print(f"❌ Processus tué : PID {pid}")
            except Exception as e:
                print(f"⚠️ Erreur lors de la tentative de suppression du PID {pid} : {e}")
    except subprocess.CalledProcessError:
        print("✅ Aucun processus Chrome/Chromedriver actif à tuer.")

# ──────────────── 1. Nettoyage d’éventuelles instances existantes ────────────────
kill_existing_chrome_sessions()

# ──────────────── 2. Définition des chemins locaux Chrome et ChromeDriver ────────────────
CHROME_PATH = "/usr/bin/google-chrome"
CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver"

# ──────────────── 3. Identifiants Proxy (exemple : IPRoyal) ────────────────
USERNAME_BASE = "Yz3XQbz7vR2z3qmo"
PASSWORD = "rUwiPZvJ8YF5tR0b"
PROXY_HOST = "geo.iproyal.com"
PROXY_PORT = 12321

# ──────────────── 4. Modèles pour le proxy et son extension ────────────────
manifest_json = """
{
    "version": "1.0.0",
    "manifest_version": 2,
    "name": "IPRoyal Proxy",
    "permissions": [
        "proxy", "tabs", "unlimitedStorage", "storage",
        "<all_urls>", "webRequest", "webRequestBlocking"
    ],
    "background": { "scripts": ["background.js"] },
    "minimum_chrome_version":"22.0.0"
}
"""

background_js_template = """
var config = {
    mode: "fixed_servers",
    rules: {
        singleProxy: {
            scheme: "http",
            host: "%s",
            port: parseInt(%s)
        },
        bypassList: ["localhost"]
    }
};
chrome.proxy.settings.set({value: config, scope: "regular"}, function() {});
function callbackFn(details) {
    return {
        authCredentials: { username: "%s", password: "%s" }
    };
}
chrome.webRequest.onAuthRequired.addListener(callbackFn, {urls: ["<all_urls>"]}, ['blocking']);
"""

# ──────────────── 5. Service ChromeDriver ────────────────
service = Service(executable_path=CHROMEDRIVER_PATH)

print("[INFO] Lancement boucle navigation...")

iteration = 0
while True:
    iteration += 1

    # ──────────────── 6. Configuration des options Chrome ────────────────
    options = Options()
    options.binary_location = CHROME_PATH
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-web-security")
    options.add_argument("--disable-webrtc")
    options.add_argument("--start-maximized")

    # 🔐 Génère un répertoire temporaire unique => évite le verrou user-data-dir
    user_data_dir = tempfile.mkdtemp(prefix="chrome_user_data_")
    options.add_argument(f"--user-data-dir={user_data_dir}")

    # User-Agent aléatoire
    ua = UserAgent()
    random_ua = ua.random
    options.add_argument(f"--user-agent={random_ua}")

    # Construction du pseudo username (session_id = agentXYZ)
    session_id = f"agent{random.randint(1000, 9999)}"
    proxy_username = f"{USERNAME_BASE}-country-ch-{session_id}"
    proxy_url = f"http://{PROXY_HOST}:{PROXY_PORT}"

    # Création de l’extension ZIP proxy (plugin)
    background_js = background_js_template % (PROXY_HOST, PROXY_PORT, proxy_username, PASSWORD)
    pluginfile = f"proxy_auth_extension_{session_id}.zip"
    with zipfile.ZipFile(pluginfile, 'w') as zp:
        zp.writestr("manifest.json", manifest_json)
        zp.writestr("background.js", background_js)

    options.add_extension(pluginfile)
    options.add_argument(f"--proxy-server={proxy_url}")

    print(f"[INFO] Itération {iteration} - Proxy : {proxy_username}@{PROXY_HOST}:{PROXY_PORT}")
    print(f"[INFO] Itération {iteration} - User-Agent : {random_ua}")

    # ──────────────── 7. Lancement du navigateur + navigation ────────────────
    try:
        driver = webdriver.Chrome(service=service, options=options)
        driver.get("https://abrahamjuliot.github.io/creepjs/")
        time.sleep(8)

        screenshot_name = f"fingerprint_{session_id}.png"
        driver.save_screenshot(screenshot_name)
        print(f"[📸] Screenshot enregistré : {screenshot_name}")

    except Exception as e:
        print(f"[❌ ERREUR] : {e}")

    finally:
        # Fermer le navigateur
        try:
            driver.quit()
        except:
            pass

        # Nettoyer l’extension et le dossier user-data-dir
        try:
            os.remove(pluginfile)
        except:
            pass
        try:
            os.rmdir(user_data_dir)
        except:
            pass

        # Pause avant la prochaine itération
        print("[⏳] Pause 2 minutes...\n")
        time.sleep(120)
