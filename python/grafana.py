import asyncio
import asyncpg
import pandas
from typing import Sequence
import re


def clean_columns(columns: Sequence[str]) -> Sequence[str]:
    cleaner = re.compile("[^a-zA-Z_]")
    clean_column_names = list(
        cleaner.sub("", x.lower().replace(" ", "_")) for x in columns
    )
    return clean_column_names


async def set_up_database(conn: asyncpg.Connection, columns: Sequence[str]) -> None:
    async with conn.transaction():
        await conn.execute("""
            DROP TABLE IF EXISTS ship_data
        """)

        column_definitions = ", ".join(f"{x} Float" for x in columns)
        table_definition = f"""
            CREATE TABLE IF NOT EXISTS ship_data (
                PRIMARY KEY (index),
                {column_definitions}
            )
        """
        print(table_definition)
        await conn.execute(table_definition)


async def push_data(data: pandas.DataFrame) -> None:
    conn = await asyncpg.connect(
        "postgresql://postgres:example@localhost:5432/postgres"
    )
    clean_cols = clean_columns(tuple(data.columns))
    await set_up_database(conn, clean_cols)

    data_tuples = [tuple(x) for x in data.values]
    await conn.copy_records_to_table(
        "ship_data", records=data_tuples, columns=clean_cols
    )
    await conn.close()


def push_data_sync(data: pandas.DataFrame) -> None:
    asyncio.run(push_data(data))
