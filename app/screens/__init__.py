"""The app's screens, one module per screen, and the pieces they share.

    shared.py     data loading, the Reader history, Browse nodes, search, related records, saved searches
    records.py    Archive home, Browse and the Reader
    assistant.py  the question box, its answer and sources
    intake.py     Add files and the Intake screen
    review.py     the Review queue

app.py loads the data on every run and passes it in with use(); each module declares the
names it reads at its top.
"""


def use(**data):
    """Give the screen modules this run's data (the archive, the queue, the taxonomy lookups)."""
    from . import assistant, intake, records, review, shared
    for module in (shared, assistant, records, intake, review):
        for name, value in data.items():
            if hasattr(module, name):
                setattr(module, name, value)
