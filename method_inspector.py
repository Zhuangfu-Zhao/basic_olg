from copy import deepcopy
import numpy as np

METHOD_IO_SPEC = {
    "_prices": {
        "inputs": [
            "KK[it]", "LL[it]", "tau_w[it]", "tau_r[it]",
            "tau_c[it]", "tau_p[it]", "kappa[it]", "w[max(it-1, 0)]"
        ],
        "outputs": [
            "r[it]", "w[it]", "wn[it]", "Rn[it]", "p[it]", "pen[it]"
        ],
    },
    "_decisions": {
        "inputs": [
            "wn[it]", "wn[min(it+1, m.TT)]",
            "Rn[it]", "Rn[min(it+1, m.TT)]", "Rn[min(it+2, m.TT)]",
            "p[it]", "p[min(it+1, m.TT)]", "p[min(it+2, m.TT)]",
            "pen[it]", "pen[min(it+1, m.TT)]", "pen[min(it+2, m.TT)]",
            "a[max(it-1, 0), 1]", "a[max(it-1, 0), 2]", "beta", "gamma"
        ],
        "outputs": [
            "c[it, :]", "a[it, :]"
        ],
    },
    "_aggregations": {
        "inputs": [
            "c[it, :]", "a[it, :]", "KK[it]", "KK[min(it+1, m.TT)]",
            "b_y[max(it-1, 0)]", "g", "n", "alpha", "delta", "LL[it]"
        ],
        "outputs": [
            "CC[it]", "AA[it]", "GG[it]", "YY[it]", "BB[it]", "KK[it]", "II[it]"
        ],
    },
    "_government": {
        "inputs": [
            "tax[it]", "r[it]", "w[it]", "CC[it]", "AA[it]", "LL[it]",
            "BB[it]", "BB[min(it+1, m.TT)]", "GG[it]",
            "tau_c[it]", "tau_w[it]", "tau_r[it]", "pen[it]", "n"
        ],
        "outputs": [
            "tau_c[it]", "tau_w[it]", "tau_r[it]", "tau_p[it]"
        ],
    },
}

def _expr_for_model(expr):
    return expr if expr.startswith("m.") else f"m.{expr}"

def _resolve_expr(m, expr, it):
    return eval(
        _expr_for_model(expr),
        {"np": np, "max": max, "min": min},
        {"m": m, "it": it},
    )

def _assign_expr(m, expr, value, it):
    exec(
        f"{_expr_for_model(expr)} = value",
        {"np": np, "max": max, "min": min},
        {"m": m, "it": it, "value": value},
    )

def _clone_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)

def _format_value(value, digits=6):
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=digits, suppress_small=False)
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    return value

def _values_equal(left, right, tol=1e-10):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.allclose(np.asarray(left), np.asarray(right), atol=tol, rtol=tol)
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(right, (int, float, np.integer, np.floating)):
        return abs(float(left) - float(right)) <= tol
    return left == right

def _snapshot(m, exprs, it):
    return {expr: _clone_value(_resolve_expr(m, expr, it)) for expr in exprs}

def _format_snapshot(snapshot):
    return {key: _format_value(value) for key, value in snapshot.items()}

def print_inspection(report):
    print(f"Method: {report['method']}   it = {report['it']}")
    if report['overrides']:
        print("Overrides applied on the copied model:")
        for key, value in report['overrides'].items():
            print(f"  {key} = {_format_value(value)}")
    else:
        print("Overrides applied on the copied model: none")

    print("\nEffective inputs read before execution:")
    for key, value in report['inputs_before'].items():
        print(f"  {key:<28} -> {value}")

    print("\nOutputs before execution:")
    for key, value in report['outputs_before'].items():
        print(f"  {key:<28} -> {value}")

    print("\nOutputs after execution:")
    for key, value in report['outputs_after'].items():
        print(f"  {key:<28} -> {value}")

    print("\nChanged outputs:")
    if report['changed_outputs']:
        for key, change in report['changed_outputs'].items():
            print(f"  {key:<28} : {change['before']}  ->  {change['after']}")
    else:
        print("  no monitored output changed")

def inspect_method(model, method_name, it, overrides=None, show=True):
    if method_name not in METHOD_IO_SPEC:
        raise ValueError(f"Unsupported method: {method_name}. Choose from {list(METHOD_IO_SPEC)}")

    m = deepcopy(model)
    overrides = overrides or {}
    for expr, value in overrides.items():
        _assign_expr(m, expr, value, it)

    spec = METHOD_IO_SPEC[method_name]
    inputs_before_raw = _snapshot(m, spec['inputs'], it)
    outputs_before_raw = _snapshot(m, spec['outputs'], it)

    getattr(m, method_name)(it)

    outputs_after_raw = _snapshot(m, spec['outputs'], it)
    changed_outputs = {}
    for expr in spec['outputs']:
        before = outputs_before_raw[expr]
        after = outputs_after_raw[expr]
        if not _values_equal(before, after):
            changed_outputs[expr] = {
                'before': _format_value(before),
                'after': _format_value(after),
            }

    report = {
        'method': method_name,
        'it': it,
        'overrides': {k: _clone_value(v) for k, v in overrides.items()},
        'inputs_before': _format_snapshot(inputs_before_raw),
        'outputs_before': _format_snapshot(outputs_before_raw),
        'outputs_after': _format_snapshot(outputs_after_raw),
        'changed_outputs': changed_outputs,
        'copied_model': m,
    }

    if show:
        print_inspection(report)

    return report

def inspect_method_on_copy(model_copy, method_name, it, show=True):
    if method_name not in METHOD_IO_SPEC:
        raise ValueError(f"Unsupported method: {method_name}. Choose from {list(METHOD_IO_SPEC)}")

    spec = METHOD_IO_SPEC[method_name]
    inputs_before_raw = _snapshot(model_copy, spec['inputs'], it)
    outputs_before_raw = _snapshot(model_copy, spec['outputs'], it)

    getattr(model_copy, method_name)(it)

    outputs_after_raw = _snapshot(model_copy, spec['outputs'], it)
    changed_outputs = {}
    for expr in spec['outputs']:
        before = outputs_before_raw[expr]
        after = outputs_after_raw[expr]
        if not _values_equal(before, after):
            changed_outputs[expr] = {
                'before': _format_value(before),
                'after': _format_value(after),
            }

    report = {
        'method': method_name,
        'it': it,
        'inputs_before': _format_snapshot(inputs_before_raw),
        'outputs_before': _format_snapshot(outputs_before_raw),
        'outputs_after': _format_snapshot(outputs_after_raw),
        'changed_outputs': changed_outputs,
        'copied_model': model_copy,
    }

    if show:
        print_inspection({**report, 'overrides': {}})

    return report

def inspect_chain(model, method_names, it, overrides=None, show=True):
    m = deepcopy(model)
    overrides = overrides or {}
    for expr, value in overrides.items():
        _assign_expr(m, expr, value, it)

    reports = []
    if show:
        print(f"Chain inspection at it = {it}")
        if overrides:
            print("Overrides applied before the chain starts:")
            for key, value in overrides.items():
                print(f"  {key} = {_format_value(value)}")
        else:
            print("Overrides applied before the chain starts: none")

    for step, method_name in enumerate(method_names, start=1):
        report = inspect_method_on_copy(m, method_name, it, show=False)
        reports.append(report)

        if show:
            print(f"\nStep {step}: {method_name}")
            if report['changed_outputs']:
                for key, change in report['changed_outputs'].items():
                    print(f"  {key:<28} : {change['before']}  ->  {change['after']}")
            else:
                print("  no monitored output changed")

    return {
        'it': it,
        'method_names': list(method_names),
        'overrides': {k: _clone_value(v) for k, v in overrides.items()},
        'reports': reports,
        'copied_model': m,
    }
