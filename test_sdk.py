import os
from foundry_local_sdk import Configuration, FoundryLocalManager

def main():
    print("Foundry Local Manager başlatılıyor...")
    cache_dir = os.path.expanduser("~/.foundry/cache/models")
    config = Configuration(app_name="rag_assistant_test", model_cache_dir=cache_dir)
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("phi-3.5-mini modeli CLI önbelleğinden yükleniyor...")
    model = manager.catalog.get_model("phi-3.5-mini")
    model.load()
    
    print("Chat client oluşturuluyor ve test mesajı gönderiliyor...")
    client = model.get_chat_client()
    response = client.complete_chat([{"role": "user", "content": "Merhaba, sen kimsin?"}])
    
    print("\n--- Model Yanıtı ---")
    print(response.choices[0].message.content)
    print("--------------------\n")
    
    model.unload()
    print("Model başarıyla bellekten temizlendi.")

if __name__ == "__main__":
    main()
