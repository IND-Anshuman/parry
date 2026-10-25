"""Print spend recorded by parry's SQLite storage, refreshing every few seconds.

Enable storage in your app (``parry.install(storage="parry.db")``), then:

    python examples/cost_dashboard.py parry.db

For a one-off report the CLI does the same: ``parry stats --db parry.db``.
"""

import sys
import time

from parry._storage import SQLiteStore, start_of_today


def render(store: SQLiteStore) -> str:
    life, today = store.spend(), store.spend(start_of_today())
    if life is None or today is None:
        return "storage unavailable"
    lines = [
        f"lifetime ${life.cost_usd:10.4f}   {life.requests:>7} requests",
        f"today    ${today.cost_usd:10.4f}   {today.requests:>7} requests",
        "",
        f"{'model':<42} {'requests':>9} {'in tok':>10} {'out tok':>10} {'cost':>11}",
    ]
    for name, s in store.spend_by_model():
        lines.append(
            f"{name:<42} {s.requests:>9} {s.input_tokens:>10} {s.output_tokens:>10} ${s.cost_usd:>10.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "parry.db"
    once = "--once" in sys.argv
    store = SQLiteStore(path)
    try:
        while True:
            print("\033[2J\033[H" if not once else "", end="")
            print(f"parry spend — {path}\n")
            print(render(store))
            if once:
                return
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        store.close()


if __name__ == "__main__":
    main()
