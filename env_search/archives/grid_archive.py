"""Custom GridArchive."""
import gin
import numpy as np
import ribs.archives
from ribs.archives._archive_base import readonly
from ribs.archives._elite import EliteBatch


@gin.configurable
class GridArchive(ribs.archives.GridArchive):
    """Based on pyribs GridArchive.

    This archive records history of its objectives and behavior values if
    record_history is True. Before each generation, call new_history_gen() to
    start recording history for that gen. new_history_gen() must be called
    before calling add() for the first time.
    """

    def __init__(self,
                 *,
                 solution_dim,
                 dims,
                 ranges,
                 learning_rate=1.0,
                 threshold_min=-np.inf,
                 epsilon=1e-6,
                 qd_score_offset=0.0,
                 seed=None,
                 dtype=np.float64,
                 record_history=True):
        super().__init__(
            solution_dim=solution_dim,
            dims=dims,
            ranges=ranges,
            learning_rate=learning_rate,
            threshold_min=threshold_min,
            epsilon=epsilon,
            qd_score_offset=qd_score_offset,
            seed=seed,
            dtype=dtype,
        )
        self._record_history = record_history
        self._history = [] if self._record_history else None

    # def best_elite(self):
    #     """Returns the best Elite in the archive."""
    #     if self.empty:
    #         raise IndexError("No elements in archive.")

    #     objectives = self._objective_values[self._occupied_indices_cols]
    #     idx = self._occupied_indices[np.argmax(objectives)]
    #     return ribs.archives.Elite(
    #         readonly(self._solution_arr[idx]),
    #         self._objective_values[idx],
    #         readonly(self._measures_arr[idx]),
    #         idx,
    #         self._metadata[idx],
    #     )

    def sample_top_elites(self, n_samples, p):
        """Sample `n_samples` elites from the top elites

        Args:
            n_samples (int): number of samples
            p (float): portion of top elites to choose from
                e.g. p=0.25 samples from top 25% of the elites, ranked by obj
        """
        # No solution, return None
        if self.empty:
            raise IndexError("No elements in archive.")

        n_candidates = int(p * self._num_occupied)

        # Not enough candidate, sample from all elites
        if n_candidates < n_samples:
            return self.sample_elites(n_samples)

        all_valid_indices = self._occupied_indices[:self._num_occupied]
        all_valid_objs = self._objective_arr[all_valid_indices]
        sorted_valid_indices = [
            x for _, x in sorted(
                zip(all_valid_objs, all_valid_indices),
                key=lambda pair: pair[0],
                reverse=True,
            )
        ]
        candidates = sorted_valid_indices[:n_candidates]

        # We should have more candidates than samples, so we sample without
        # replacement (elites do not repeat) to promote exploration
        selected_indices = self._rng.choice(candidates,
                                            size=n_samples,
                                            replace=False)

        return EliteBatch(
            readonly(self._solution_arr[selected_indices]),
            readonly(self._objective_arr[selected_indices]),
            readonly(self._measures_arr[selected_indices]),
            readonly(selected_indices),
            readonly(self._metadata_arr[selected_indices]),
        )

    def new_history_gen(self):
        """Starts a new generation in the history."""
        if self._record_history:
            self._history.append([])

    def history(self):
        """Gets the current history."""
        return self._history

    def add(self,
            solution_batch,
            objective_batch,
            measures_batch,
            metadata_batch=None):
        status_batch, val_batch = super().add(
            solution_batch,
            objective_batch,
            measures_batch,
            metadata_batch,
        )

        # Only save obj and BCs in the history.
        for i, status in enumerate(status_batch):
            if self._record_history and status:
                self._history[-1].append(
                    [objective_batch[i], measures_batch[i]])

        return status_batch, val_batch

    def add_single(self, solution, objective, measures, metadata=None):
        status, value = super().add_single(
            solution,
            objective,
            measures,
            metadata=metadata,
        )

        # Only save obj and BCs in the history.
        if self._record_history and status:
            self._history[-1].append([objective, measures])

        return status, value
