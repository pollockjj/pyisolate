from typing import Any

import pytest


@pytest.mark.asyncio
async def test_extension_lifecycle(reference_host: Any) -> None:
    ext = reference_host.load_test_extension("lifecycle_test", isolated=True)

    proxy = ext.get_proxy()
    response = await proxy.ping()
    assert response == "pong"

    child_env = await proxy.get_env_var("PYISOLATE_CHILD")
    assert child_env == "1"


@pytest.mark.asyncio
async def test_non_isolated_lifecycle(reference_host: Any) -> None:


    ext = reference_host.load_test_extension("no_torch_share", isolated=True, share_torch=False)
    proxy = ext.get_proxy()
    assert await proxy.ping() == "pong"
