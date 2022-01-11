import ray


class TaskPool(object):
    """Helper class for tracking the status of many in-flight actor tasks."""

    def __init__(self):
        self._tasks = {}
        self._objects = {}
        self._fetching = []

    def add(self, worker, all_obj_ids):  #
        if isinstance(all_obj_ids, list):
            obj_id = all_obj_ids[0]
        else:
            obj_id = all_obj_ids
        self._tasks[obj_id] = worker
        self._objects[obj_id] = all_obj_ids

    def completed(self, blocking_wait=False, num=1):   #
        pending = list(self._tasks)
        # print('pending = ', pending)
        if pending:
            # print('come in pending')
            ready, _ = ray.wait(pending, num_returns=num, timeout=10.0)
            # print('ready = ', ready)
            if not ready and blocking_wait:
                # print('not ready')
                ready, _ = ray.wait(pending, num_returns=num, timeout=10.0)
            for obj_id in ready:
                # print('ready')
                yield self._tasks.pop(obj_id), self._objects.pop(obj_id)

    @property
    def count(self):
        return len(self._tasks)
