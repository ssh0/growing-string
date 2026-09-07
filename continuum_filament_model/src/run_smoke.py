"""Small non-GUI smoke run for the new continuum prototype."""

from __future__ import annotations

import numpy as np

from growing_filament.model import FilamentState, ModelParameters, OverdampedGrowingFilament
from growing_filament.observables import summary


def main() -> None:
    positions = np.asarray(
        [[0.0, 0.0], [1.0, 0.25], [2.0, 0.5], [3.0, 0.25], [4.0, 0.0]],
        dtype=float,
    )
    state = FilamentState(positions, np.ones(4))
    parameters = ModelParameters(
        axial_stiffness=100.0,
        bending_stiffness=0.5,
        drag_density=1.0,
        growth_rate=0.02,
        reference_length=1.0,
        dt=1.0e-4,
        t_end=0.01,
        a_max=1.2,
        fixed_left=True,
        fixed_right=True,
    )
    model = OverdampedGrowingFilament(state, parameters)
    trajectory = model.run()
    print("states:", len(trajectory))
    print("initial:", summary(trajectory[0]))
    print("final:", summary(trajectory[-1]))
    print("accepted_steps:", model.accepted_steps)
    print("rejected_steps:", model.rejected_steps)


if __name__ == "__main__":
    main()
