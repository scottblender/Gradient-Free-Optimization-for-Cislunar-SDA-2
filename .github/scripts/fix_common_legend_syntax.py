from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"Expected one match in {path}, found {n}: {old!r}")
    p.write_text(text.replace(old, new, 1))

p = 'scripts/make_reviewer2_curated_figures.m'
replace_once(
    p,
    '    missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];\n\n    observerSpecs = { ...\n    export_shared_result_legend(out,"baseline_observer_metric_legend", ...\n',
    '    missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];\n\n    export_shared_result_legend(out,"baseline_observer_metric_legend", ...\n',
)
replace_once(
    p,
    '    missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];\n\n    % Screening ON/OFF: only the physical metrics requested for the paper.\n    export_shared_result_legend(out,"ga_screening_metric_legend", ...\n',
    '    missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];\n\n    export_shared_result_legend(out,"ga_screening_metric_legend", ...\n',
)

p = 'scripts/plot_parallel_speed.m'
replace_once(
    p,
    'end\n% Keep the numeric printout beside the final figures as well as in the raw run.\nexport_shared_result_legend(outputDirectory,"parallel_speed_lg_legend", ...\n',
    'end\nexport_shared_result_legend(outputDirectory,"parallel_speed_lg_legend", ...\n',
)

print('Fixed common-legend patch syntax and comments.')
