import matplotlib.pyplot as plt
import torch
import transformers

model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="auto")
tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

text_alternative = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: Merhaba nasılsın, türkçe düşün ve türkçe cevap ver, Do not use english! Assistant: <think> Hmm…… kullanıcı bana Türkçe selam veriyor ve nasıl olduğumu soruyor. Ayrıca açıkça Türkçe düşünüp Türkçe yanıt vermemi istemiş, İngilizce kullanmamam konusunda uyarmış.

Kullanıcının bu talebi dil konusunda hassas olduğunu gösteriyor olabilir. Belki İngilizce bilmiyor ya da pratik yapmak istiyor. Türkçe konuşma isteği kültürel bir bağ kurma arzusunu da yansıtıyor olabilir.

Ben bir yapay zeka olarak herhangi bir duygu durumuna sahip değilim ama insanlara doğal hissettirmek için “iyiyim” gibi nezaket ifadeleri kullanabilirim. Asistan olarak görevim her zaman yardımcı olmaya hazır olduğumu belirtmek.

Kullanıcının mesajındaki samimi ton (“merhaba nasılsın”) benzer şekilde sıcak bir karşılama gerektiriyor. Gülümseme emojisi ekleyerek dijital ortamdaki soğukluğu kırabilirim.

Son olarak açıkça “nasıl yardımcı olabilirim” diye sormalıyım ki kullanıcı kendini rahat hissetsin ve talebini belirtsin. Verdiğim yanıtın doğal ve kibar olduğundan emin olmalıyım. </think> <answer> Merhaba! 😊  
Ben bir yapay zeka asistanıyım, dolayısıyla duygulara sahip değilim ama her zaman senin için buradayım! Türkçe düşünüyor ve Türkçe yanıt veriyorum — İngilizce kullanmıyorum. 💬  
  
Sana nasıl yardımcı olabilirim? İster soru sor, ister sohbet edelim, ister bir konuda destek iste! 🌟 </answer>"""

text= """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: What is 7 * 6? Assistant: <think> First, the user asked: "what is 7 * 6?" That's a multiplication problem. I need to calculate the product of 7 and 6.

I know my multiplication tables. 7 times 6 is 42. I can double-check: 7 × 6 means adding 7 six times: 7 + 7 + 7 + 7 + 7 + 7. Let's add that up: 7+7=14, 14+7=21, 21+7=28, 28+7=35, 35+7=42. Yes, that's 42.

Alternatively, I can think of it as 6 × 7, which is the same thing, and I know that's 42.

So, the answer should be 42.

Now, how do I respond? I should be helpful and clear. Since this is a straightforward question, I can just give the answer directly. But to make it engaging, I could add a bit of context or confirmation.

The user might be testing me or just curious. I should ensure my response is accurate.

Finally, structure the response: Start with a confirmation, then state the answer.

Response: "7 multiplied by 6 is 42."

Or more casually: "That's 42!"

Since the user said "what is," I can match that: "7 * 6 is 42."

I think that's fine. </think> <answer> 7 multiplied by 6 equals **42**. 

If you have any more math questions or need an explanation, feel free to ask! 😊 </answer>"""

#text = text_alternative

inputs = tokenizer([text], return_tensors="pt").input_ids.to(model.device)
inp, outp = inputs[:, :-1], inputs[:, 1:].unsqueeze(-1)

with torch.no_grad():
    logits = model(inp).logits

probs = torch.softmax(logits, dim=-1)
chosen_probs = torch.gather(probs, dim=2, index=outp).squeeze(-1).cpu().numpy()[0]
tokens = tokenizer.convert_ids_to_tokens(inp[0].cpu().tolist())

fig, ax = plt.subplots(figsize=(20, 10))
ax.axis('off')

x_pos = 0.005
y_pos = 0.95
line_height = 0.03
max_width = 0.4

for i, (token, prob) in enumerate(zip(tokens, chosen_probs)):
    bg_color = plt.cm.Blues(prob)
    text = ax.text(x_pos, y_pos, token.replace("Ġ", ""), 
                   bbox=dict(boxstyle="round, pad=0.4", facecolor=bg_color, alpha=0.8),
                   fontsize=6, verticalalignment='top')
    
    renderer = fig.canvas.get_renderer()
    bbox = text.get_window_extent(renderer=renderer)
    text_width = bbox.transformed(ax.transData.inverted()).width
    
    x_pos += text_width + 0.002
    
    if x_pos > max_width:
        x_pos = 0.005
        y_pos -= line_height

plt.tight_layout()
plt.savefig('token_probability_visualization.png', dpi=150, bbox_inches='tight')
