def arithmetic_tool(op, a, b):
    try:
        a, b = float(a), float(b)
        if op == 'add': return a + b
        if op == 'subtract': return a - b
        if op == 'multiply': return a * b
        if op == 'divide': return a / b if b != 0 else "Error: Div0"
    except: return "Error: Invalid numbers"
    return "Error: Unknown Op"
