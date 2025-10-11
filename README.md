# mini-constitutional-classifier
a reproducible pipeline that: (1) generates N candidate responses; (2) critiques each candidate vs a constitution; (3) picks (or re-ranks) the best answer; (4) produces evaluation metrics (harmlessness, helpfulness proxy)


To start, setup your tech with the following commands:
```
python -m venv venv && source venv/bin/activate
pip install -U pip
pip install transformers datasets peft accelerate bitsandbytes evaluate streamlit flask sentence-transformers matplotlib fire
```

