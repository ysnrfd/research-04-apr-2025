from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# نام مدل
model_name = "HuggingFaceTB/SmolLM2-135M"

def main():
    # بارگذاری مدل و توکنایزر
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # تنظیم pad_token برای جلوگیری از خطا
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # انتقال مدل به CPU
    device = torch.device("cpu")
    model.to(device)
    
    # بارگذاری مجموعه داده از Hugging Face
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # تابع پردازش داده‌ها
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    # پردازش داده‌ها
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # تنظیمات آموزش برای اجرا روی CPU
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=0.01,
        per_device_train_batch_size=2,  # کاهش batch size برای کاهش مصرف RAM
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
        save_steps=1000,
        fp16=False,  # غیرفعال کردن fp16 چون روی CPU پشتیبانی نمی‌شود
        no_cuda=True,  # غیرفعال کردن استفاده از GPU
    )
    
    # ایجاد data collator برای مدل‌های زبانی خودبازگشتی
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # چون مدل causal است، Masked Language Modeling خاموش است
    )
    
    # ایجاد Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # شروع آموزش
    trainer.train()
    
    # ذخیره مدل نهایی
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

if __name__ == "__main__":
    main()