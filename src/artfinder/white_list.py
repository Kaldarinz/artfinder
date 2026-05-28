# SPDX-FileCopyrightText: 2025-present Anton Popov <a.popov.fizteh@gmail.com>
#
# SPDX-License-Identifier: MIT
"""
Classes for handling Russian white list of journals.
"""

import asyncio
import logging
import threading
import aiohttp
from artfinder.scimagojr import SciMagoJR


async def get_journal_info(issn: str) -> int | None:
    """
    Get journal information from the Russian white list.

    Parameters
    -----------
        issn: The ISSN of the journal. Should be in format '12345678' or '12345678X'

    Returns
    -------
        A dictionary containing the journal information, or None if not found.
    """

    url = f"https://journalrank.rcsi.science/api/record-sources/{issn}/level"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    levels = [key for key in data.keys() if key.startswith("level_")]
                    if not levels:
                        logging.warning(
                            f"No level information found for journal with ISSN {issn}."
                        )
                        return None
                    levels.sort(key=lambda x: x.split("_")[-1], reverse=True)
                    while levels:
                        level_key = levels.pop(0)
                        try:
                            level_value = int(data[level_key])
                            return level_value
                        except (ValueError, TypeError):
                            continue
                    return None

                else:
                    logging.warning(
                        f"Journal with ISSN {issn} not found in white list."
                    )
                    return None
        except aiohttp.ClientError as e:
            logging.error(f"Error fetching journal info for ISSN {issn}: {e}")
            return None


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, dict | None] = {}
    error: dict[str, Exception] = {}

    def runner():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:
            error["value"] = exc

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]

    return result.get("value")


def get_journal_info_sync(
    title: str | None = None, issn: str | None = None
) -> dict | None:
    """
    Synchronous version of get_journal_info for use in non-async contexts.
    """
    if issn is None and title is None:
        logging.error("Either title or issn must be provided.")
        return None
    if issn is None:
        scimago = SciMagoJR()
        journal_data = scimago.get_journal(title=title)
        if journal_data is not None:
            issn_field = journal_data["issns"]
            if isinstance(issn_field, str):
                issns = [item.strip() for item in issn_field.split(",") if item.strip()]
            else:
                issns = [str(item).strip() for item in issn_field if str(item).strip()]
        else:
            logging.warning(f"Journal with title '{title}' not found in SciMagoJR.")
            return None
    else:
        issns = [issn]

    async def _get_first_journal_info() -> dict | None:
        results = await asyncio.gather(*(get_journal_info(issn_) for issn_ in issns))
        for journal_info in results:
            if journal_info is not None:
                return journal_info
        return None

    return _run_coro_sync(_get_first_journal_info())
