import re

with open(r'c:\Users\user\Projects\stagewahch\frontend\index.html', 'r', encoding='utf-8') as f:
    content = f.read()

new_content = re.sub(
    r'<div class="logo">.*?</div>', 
    '<div class="logo">\n      <img src="olive.png" alt="Logo">\n    </div>', 
    content, 
    flags=re.DOTALL
)

with open(r'c:\Users\user\Projects\stagewahch\frontend\index.html', 'w', encoding='utf-8') as f:
    f.write(new_content)
print("Done!")
