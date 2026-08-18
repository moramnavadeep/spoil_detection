# Contributing to NutriFresh AI

Thank you for your interest in contributing to **NutriFresh AI**! We welcome improvements to the deep learning models, API services, preprocessor optimizations, and documentation.

---

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/nutrifresh-ai.git
   cd nutrifresh-ai
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run tests to verify setup:**
   ```bash
   pytest tests/ -v
   ```

---

## 🌿 Contribution Guidelines

1. **Branching Model**: Use feature branches off `main` (e.g. `feat/resnet-backbone`, `fix/upload-validation`).
2. **Coding Standards**:
   - Write type-annotated, PEP 8 compliant code.
   - Include docstrings on all public methods and functions.
   - Ensure all tests pass before submitting a Pull Request.
3. **Commit Convention**: Follow Conventional Commits:
   - `feat: add model evaluation metrics export`
   - `fix: correct image normalization scale`
   - `test: add batch inference edge cases`
   - `docs: update API documentation`

---

## 🧪 Testing

Always run the full test suite before pushing:
```bash
pytest tests/ -v --cov=src
```
