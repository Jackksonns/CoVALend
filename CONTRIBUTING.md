# Contributing to CoVALend

Thank you for your interest in contributing to CoVALend! This document provides guidelines for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)

## 📜 Code of Conduct

This project adheres to responsible AI principles and ethical research practices. By participating, you agree to:

- Use the code only for academic research purposes
- Not use it for individual credit assessment or commercial lending decisions
- Respect data privacy and fairness considerations
- Follow all applicable laws and regulations

## 🤝 How Can I Contribute?

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU, etc.)

### Suggesting Enhancements

We welcome suggestions for:
- New features
- Performance improvements
- Documentation improvements
- Additional fairness metrics

### Code Contributions

Areas where contributions are especially welcome:
- Training scripts for Stage 1 and Stage 2
- Data preprocessing utilities
- Visualization tools
- Unit tests
- Documentation

## 🛠️ Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CoVALend.git
   cd CoVALend
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

5. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 Coding Standards

### Python Style Guide

- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and modular

### Docstring Format

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description of the function
    
    Longer description if needed explaining the algorithm,
    implementation details, or important notes.
    
    Args:
        param1 (Type1): Description of param1
        param2 (Type2): Description of param2
        
    Returns:
        ReturnType: Description of return value
        
    Raises:
        ValueError: When param1 is invalid
        
    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
        Expected output
    """
    pass
```

### Code Organization

- Place model components in `covalend/models/`
- Place evaluation code in `covalend/evaluation/`
- Place data processing in `covalend/data/`
- Place utilities in `covalend/utils/`

### Testing

- Write unit tests for new functionality
- Place tests in `tests/` directory
- Run tests before submitting:
  ```bash
  pytest tests/
  ```

## 📤 Submitting Changes

### Pull Request Process

1. Update documentation for any new features
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md with your changes
5. Submit pull request with clear description

### Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with main
- [ ] No merge conflicts

### Commit Message Format

```
type(scope): brief description

Detailed explanation of changes (optional)

Fixes #issue_number (if applicable)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Build process or auxiliary tool changes

Example:
```
feat(evaluation): add precision-recall curve plotting

Adds visualization function for precision-recall curves
in the evaluation module. Includes example usage in docs.

Fixes #42
```

## 🔍 Code Review Process

- At least one maintainer will review your PR
- Address feedback and make requested changes
- Once approved, a maintainer will merge your PR

## ❓ Questions?

If you have questions about contributing, please:
- Check existing issues and discussions
- Open a new issue with the "question" label
- Contact the maintainers (see README for contact info)

## 🙏 Thank You!

Your contributions help make CoVALend better for the research community!

---

© 2026 CoVALend Team
