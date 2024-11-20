# yeni-yeni-depolar

KEYLER
############################################################
!pip install groq together openai
import os
import time
from openai import OpenAI
from groq import Groq
import requests
from together import Together


# Set the API key for the OPENROUTER API
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual OpenAI key
# Set the API key for the NVIDIA API
nvidia_api_key = "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"
# Set the API key for OpenRouter API
os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "sk-or-v1-d28f31a29084d0a4a0f0dd6471ed3088cee2ec994bc9daf141fe85fff5c9e0**"  # Replace with your actual OpenAI key
os.environ["TOGETHER_API_KEY"] = "91f4c789e41cb9dff56b44ba43ec90da2cdba677d1f1752fd2e26152f1deb453"



'''
GROQ API
'''
# Initialize the Groq client
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Measure time for Groq API
start_time_groq = time.time()

# Create a chat completion for Groq
groq_completion = groq_client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Write e-mail regex with python",
        }
    ],
    model="llama3-8b-8192",
)

# Print the result for Groq API
print(groq_completion.choices[0].message.content)

end_time_groq = time.time()
execution_time_groq = end_time_groq - start_time_groq


'''
NVIDIA API
'''

# NVIDIA API URL'sini belirtin
nvidia_api_url = "https://integrate.api.nvidia.com/v1/chat/completions"

# Zaman ölçümünü başlat
start_time_nvidia = time.time()

# API çağrısı için istek verilerini hazırlayın
data = {
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [{"role": "user", "content": "Write python regex for e-mail."}],
    "temperature": 0.2,
    "max_tokens": 1024,
    "stream": False  # Stream desteği yoksa bunu False yapın
}

# İstek başlıklarını belirtin
headers = {
    "Authorization": f"Bearer {nvidia_api_key}",
    "Content-Type": "application/json"
}

# NVIDIA API'ye POST isteği gönder
response = requests.post(nvidia_api_url, headers=headers, json=data)

# Yanıtı kontrol et ve yazdır
if response.status_code == 200:
    response_data = response.json()
    if "choices" in response_data:
        for choice in response_data["choices"]:
            print(choice["message"]["content"])
else:
    print(f"Error: {response.status_code}, {response.text}")

# Zaman ölçümünü bitir
end_time_nvidia = time.time()
execution_time_nvidia = end_time_nvidia - start_time_nvidia

'''
OPENROUTER API
'''

# Initialize OpenAI client for OpenRouter API
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)

# Measure time for OpenRouter API
start_time_openrouter = time.time()

try:
    # Create a chat completion for OpenRouter
    completion_openrouter = openrouter_client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "YOUR_SITE_URL",  # Optional
            "X-Title": "YOUR_APP_NAME",        # Optional
        },
        model="meta-llama/llama-3.1-405b-instruct:free",
        messages=[{"role": "user", "content": "Write e-mail regex with python"}]
    )
    
    # Print the result for OpenRouter API
    print(completion_openrouter.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")

end_time_openrouter = time.time()
execution_time_openrouter = end_time_openrouter - start_time_openrouter



# Initialize Together client
together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

# Measure time for Together API
start_time_together = time.time()
try:
    together_response = together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[{"role": "user", "content": "write regex email python"}],
        max_tokens=None,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"],
        safety_model="meta-llama/Meta-Llama-Guard-3-8B"
    )
    print("Together Response:", together_response.choices[0].message.content)

except Exception as e:
    print(f"Together Error: {e}")

end_time_together = time.time()
execution_time_together = end_time_together - start_time_together

# Print execution times
print(f"\nGROQ API Execution Time: {execution_time_groq:.2f} seconds")
print(f"\nTogether API Execution Time: {execution_time_together:.2f} seconds")
print(f"\nOpenRouter API Execution Time: {execution_time_openrouter:.2f} seconds")
print(f"\nNVIDIA API Execution Time: {execution_time_nvidia:.2f} seconds")
########################################################################################
KLONLANMAMIŞ KEYLER 
#########################################################################################33
!pip install groq together openai
import os
import time
from openai import OpenAI
from groq import Groq
from together import Together
# Set the API key for the OPENROUTER API
os.environ["GROQ_API_KEY"] = "gsk_ZonjYIIUHmOxbGrCOaFyWGdyb3FYLtn8SKzMuPxKh9xk6fq4uQ**"  # Replace with your actual OpenAI key
# Set the API key for the NVIDIA API
nvidia_api_key = "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**"
# Set the API key for OpenRouter API
os.environ["OPENAI_API_KEY"] = "sk-or-v1-f514471e03abae4c1eed7412a88ff0b20f375baf30dae914e9e571b8160ac9**"
#os.environ["OPENAI_API_KEY"] = "sk-or-v1-d28f31a29084d0a4a0f0dd6471ed3088cee2ec994bc9daf141fe85fff5c9e0**"  # Replace with your actual OpenAI key
os.environ["TOGETHER_API_KEY"] = "f7f1989e199852e8ad53b2798ad669b9f8ff1ad4e57f9b3162d9a8e57aab49**"
'''
GROQ API
'''
# Initialize the Groq client
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
# Measure time for Groq API
start_time_groq = time.time()
# Create a chat completion for Groq
groq_completion = groq_client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Write e-mail regex with python",
        }
    ],
    model="llama3-8b-8192",
)
# Print the result for Groq API
print(groq_completion.choices[0].message.content)
end_time_groq = time.time()
execution_time_groq = end_time_groq - start_time_groq
'''
NVIDIA API
'''
# Initialize the OpenAI client for NVIDIA API
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_api_key,
)
# Measure time for NVIDIA API
start_time_nvidia = time.time()
completion_nvidia = nvidia_client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "Write python regex for e-mail."}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    stream=True
)
# Print NVIDIA API response
for chunk in completion_nvidia:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
end_time_nvidia = time.time()
execution_time_nvidia = end_time_nvidia - start_time_nvidia
'''
OPENROUTER API
'''
# Initialize OpenAI client for OpenRouter API
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)
# Measure time for OpenRouter API
start_time_openrouter = time.time()
try:
    # Create a chat completion for OpenRouter
    completion_openrouter = openrouter_client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "YOUR_SITE_URL",  # Optional
            "X-Title": "YOUR_APP_NAME",        # Optional
        },
        model="meta-llama/llama-3.1-405b-instruct:free",
        messages=[{"role": "user", "content": "Write e-mail regex with python"}]
    )
    
    # Print the result for OpenRouter API
    print(completion_openrouter.choices[0].message.content)
except Exception as e:
    print(f"An error occurred: {e}")
end_time_openrouter = time.time()
execution_time_openrouter = end_time_openrouter - start_time_openrouter
# Initialize Together client
together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
# Measure time for Together API
start_time_together = time.time()
try:
    together_response = together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[{"role": "user", "content": "write regex email python"}],
        max_tokens=None,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"],
        safety_model="meta-llama/Meta-Llama-Guard-3-8B"
    )
    print("Together Response:", together_response.choices[0].message.content)
except Exception as e:
    print(f"Together Error: {e}")
end_time_together = time.time()
execution_time_together = end_time_together - start_time_together
# Print execution times
print(f"\nGROQ API Execution Time: {execution_time_groq:.2f} seconds")
print(f"\nTogether API Execution Time: {execution_time_together:.2f} seconds")
print(f"\nOpenRouter API Execution Time: {execution_time_openrouter:.2f} seconds")
print(f"\nNVIDIA API Execution Time: {execution_time_nvidia:.2f} seconds")
###################################################################################
Tİkinter kodları 
#################################################################################
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import subprocess
import re
import requests
import json
class QuestionGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Tabanlı Soru Üretici")
        self.root.geometry("700x700")
        # Model Seçimi Bölümü
        model_frame = ttk.LabelFrame(root, text="Modeller")
        model_frame.pack(fill="x", padx=10, pady=5)
        self.models = self.get_models()
        self.selected_model = tk.StringVar()
        if self.models:
            self.selected_model.set(self.models[0])
        else:
            self.selected_model.set("Model bulunamadı")
        self.model_dropdown = ttk.OptionMenu(model_frame, self.selected_model, *self.models)
        self.model_dropdown.pack(padx=10, pady=10)
        # Alan Seçimi Bölümü
        field_frame = ttk.LabelFrame(root, text="Alan Seçimi")
        field_frame.pack(fill="x", padx=10, pady=5)
        self.selected_field = tk.StringVar()
        fields = ["Dijital Dönüşüm", "Python Kodlama", "Power BI", "Ağ ve Güvenlik"]
        self.selected_field.set(fields[0])
        field_dropdown = ttk.OptionMenu(field_frame, self.selected_field, *fields)
        field_dropdown.pack(padx=10, pady=10)
        # Soru Üretme Butonu
        generate_button = ttk.Button(root, text="Soruları Üret", command=self.start_generation)
        generate_button.pack(pady=10)
        # Yükleme Çubuğu ve Süre
        progress_frame = ttk.Frame(root)
        progress_frame.pack(fill="x", padx=10, pady=5)
        self.progress = ttk.Progressbar(progress_frame, orient='horizontal', mode='determinate')
        self.progress.pack(fill="x", padx=10, pady=5)
        self.time_label = ttk.Label(progress_frame, text="Geçen Süre: 0s")
        self.time_label.pack(pady=5)
        # Çıktı Metin Kutusu
        output_frame = ttk.LabelFrame(root, text="Üretilen Sorular")
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.output_text = tk.Text(output_frame, wrap="word")
        self.output_text.pack(fill="both", expand=True, padx=10, pady=10)
    def get_models(self):
        try:
            # "ollama list" komutunu çalıştır ve çıktısını al
            result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            output = result.stdout
            # Çıktıyı parse et
            models = self.parse_ollama_list(output)
            if not models:
                messagebox.showwarning("Uyarı", "Hiç model bulunamadı.")
            return models
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Hata", f"Ollama komutu çalıştırılamadı:\n{e.stderr}")
            return []
        except FileNotFoundError:
            messagebox.showerror("Hata", "Ollama komutu bulunamadı. Lütfen Ollama'nın kurulu ve PATH'e ekli olduğundan emin olun.")
            return []
    def parse_ollama_list(self, output):
        models = []
        lines = output.strip().split('\n')
        # Başlık satırını atla
        for line in lines[1:]:
            # Satır formatı:
            # NAME                                        ID              SIZE      MODIFIED     
            # Örneğin:
            # llama3.2:latest                             a80c4f17acd5    2.0 GB    6 days ago      
            # Modellerin isimlerini al
            match = re.match(r'^(\S+)', line)
            if match:
                model_name = match.group(1)
                models.append(model_name)
        return models
    def start_generation(self):
        if not self.models:
            messagebox.showwarning("Uyarı", "Hiç model bulunamadı.")
            return
        selected_model = self.selected_model.get()
        selected_field = self.selected_field.get()
        # Soru üretme işlemini ayrı bir thread'de başlat
        threading.Thread(target=self.generate_questions, args=(selected_model, selected_field), daemon=True).start()
    def generate_questions(self, model, field):
        # Progress bar'ı determinate moda al ve sıfırla
        self.progress.config(mode='determinate', maximum=100, value=0)
        self.time_label.config(text="Geçen Süre: 0s")
        self.output_text.delete(1.0, tk.END)
        start_time = time.time()
        try:
            # Soru üretme işlemi için API çağrısı yap
            questions = self.create_questions_with_ollama(model, field)
            elapsed = int(time.time() - start_time)
            self.time_label.config(text=f"Geçen Süre: {elapsed}s")
            # Progress bar'ı tamamlanmış olarak ayarla
            self.progress['value'] = 100
            # Soruları metin kutusuna ekle
            for q in questions:
                self.output_text.insert(tk.END, f"- {q}\n")
        except Exception as e:
            # Hata durumunda progress bar'ı durdur ve hata mesajı göster
            self.progress['value'] = 0
            messagebox.showerror("Hata", f"Soru üretme sırasında bir hata oluştu:\n{e}")
    def create_questions_with_ollama(self, model, field):
        """
        Ollama API kullanarak seçilen model ve alana göre iki soru üretir.
        Args:
            model (str): Kullanılacak modelin adı.
            field (str): Soru üretilecek alan.
        Returns:
            list: Üretilen soruların listesi.
        Raises:
            Exception: API çağrısı başarısız olursa.
        """
        # Ollama API endpoint'i
        api_url = "http://localhost:11434/api/generate"  # Doğru endpoint
        # API için gerekli payload
        # Prompt'u daha net yapmak için her sorunun ayrı satırda olmasını talep ediyoruz
        payload = {
            "model": model,
            "prompt": f"Lütfen {field} alanında iki adet özgün soru üretiniz. Her bir soruyu '1.' ve '2.' ile numaralandırarak ayrı satırlarda yazınız."
        }
        headers = {
            "Content-Type": "application/json"
            # Gerekli ise yetkilendirme başlıkları ekleyin
        }
        # Streaming yanıt için stream=True
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), stream=True, timeout=60)
        # Yanıtı yazdırarak kontrol edin
        print("API Response Status Code:", response.status_code)
        # print("API Response Text:", response.text)  # Bu, çok fazla çıktı verebilir
        if response.status_code != 200:
            raise Exception(f"API çağrısı başarısız oldu. Status Code: {response.status_code}, Mesaj: {response.text}")
        generated_text = ""
        questions = []
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        json_obj = json.loads(line.decode('utf-8'))
                        response_text = json_obj.get("response", "")
                        done = json_obj.get("done", False)
                        generated_text += response_text
                        # Debug: Her bir parça ekrana yazdırılabilir
                        print(f"Received chunk: {response_text}")
                        if done:
                            break
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        continue
            if not generated_text:
                raise Exception("API'den geçerli bir yanıt alınamadı.")
            # Soruları ayırmak için numaralandırmayı kullan
            # Örneğin: "1. Soru?\n2. Soru?"
            # Regex ile "1. " ve "2. " başlangıçlarını bulup ayırabiliriz
            pattern = r'1\.\s*(.*?)\s*2\.\s*(.*)'
            match = re.search(pattern, generated_text, re.DOTALL)
            if match:
                question1 = match.group(1).strip()
                question2 = match.group(2).strip()
                questions = [question1, question2]
            else:
                # Eğer numaralandırma yoksa, satır satır ayırmayı deneyin
                questions = [q.strip() for q in generated_text.split('\n') if q.strip()]
                if len(questions) < 2:
                    raise Exception("İki soru alınamadı. API'nin yanıt formatını kontrol edin.")
            return questions[:2]  # İlk iki soruyu döndür
        except requests.exceptions.RequestException as e:
            raise Exception(f"API çağrısı sırasında bir hata oluştu: {e}")
        except Exception as e:
            raise Exception(f"Yanıt işleme sırasında bir hata oluştu: {e}")
if __name__ == "__main__":
    root = tk.Tk()
    app = QuestionGeneratorApp(root)
    root.mainloop()
    ###############################################################################################
    GRADİO SORU ÜRETCİ 
    ################################################################################################
    # !pip install gradio
import gradio as gr
# Initialize the list of tasks
tasks = []
# Function to add a task
def add_task(task, task_date):
    if task.strip():
        tasks.append({"task": task.strip(), "date": task_date})
    return update_task_list()
# Function to delete a task
def delete_task(index):
    if 0 <= index < len(tasks):
        del tasks[index]
    return update_task_list()
# Function to update the task list display
def update_task_list():
    return "\n".join([f"{i+1}. {task['task']} - Due: {task['date']}" for i, task in enumerate(tasks)])
# Create Gradio interface
with gr.Blocks(theme=gr.themes.Default(primary_hue="purple")) as demo:
    gr.Markdown("# TODO List")
    
    with gr.Row():
        task_input = gr.Textbox(label="Add Task", placeholder="Enter a new task...")
        task_date_input = gr.Textbox(label="Due Date (YYYY-MM-DD)", placeholder="Enter a date...")
    
    add_button = gr.Button("Add Task")
    task_list_display = gr.Textbox(label="Tasks", interactive=False, lines=10)
    
    with gr.Row():
        delete_index_input = gr.Number(label="Delete Task Index", precision=0)
        delete_button = gr.Button("Delete Task")
    
    # Event handlers
    add_button.click(fn=add_task, inputs=[task_input, task_date_input], outputs=task_list_display)
    delete_button.click(fn=delete_task, inputs=[delete_index_input], outputs=task_list_display)
# Launch the interface
demo.launch()
##############################################################################################
COBANOV SUBMARİZER
##############################################################################################
do mkdir YZ3
sudo chmod 777 YZ3/
cd YZ3/
git clone https://github.com/cobanov/easy-web-summarizer.git
cd ..
python3 -m venv YZ3/
source YZ3/bin/activate
cd YZ3/
cd easy-web-summarizer/
ls -la
pip install -r requirements.txt 
python3 app/webui.py
################################################################################################
