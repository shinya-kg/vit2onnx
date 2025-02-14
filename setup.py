from setuptools import setup, find_packages

setup(
    name="vit2onnx_tool",  
    version="0.1.0",  
    description="A local tool for transrate ONNX models",  
    author="koga_shinya",  
    packages=find_packages(include=["tools", "tools.*"]),  
    install_requires=[  
        "onnx",
        "onnxruntime",
        "torch",
        "numpy"
    ],
    entry_points={  
        "console_scripts": [
            "inference=tools.inference:main",  
            # ↑ `scripts/optimize_onnx.py` の `main()` を実行
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  
)