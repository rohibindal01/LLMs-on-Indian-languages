from setuptools import setup, find_packages

setup(
    name="indic-llm-eval",
    version="0.1.0",
    description="Benchmarking LLMs on Indian languages",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/your-username/indic-llm-eval",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "torch>=2.1.0",
        "huggingface_hub>=0.22.0",
        "evaluate>=0.4.1",
        "rouge-score>=0.1.2",
        "sacrebleu>=2.4.0",
        "click>=8.1.7",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "pandas>=2.2.0",
        "tabulate>=0.9.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "gemini": ["google-generativeai>=0.5.0"],
        "groq":   ["groq>=0.5.0"],
        "ollama": ["ollama>=0.1.8"],
        "all":    ["google-generativeai>=0.5.0", "groq>=0.5.0", "ollama>=0.1.8"],
    },
    entry_points={
        "console_scripts": [
            "indic-eval=indic_eval.cli:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: Hindi",
    ],
)
