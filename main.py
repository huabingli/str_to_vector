import typer
from fastapi import FastAPI

from api import router
from core.config import settings
from core.exceptions import exception_handler
from core.lifespan_handler import lifespan
from core.log import setup_logging
from core.middleware import add_middleware

setup_logging()

app = FastAPI(lifespan=lifespan)
add_middleware(app)
exception_handler(app)
app.include_router(router=router)


def run(
        port: int = typer.Option(
                8000,
                "--port",
                "-p",
                help="端口号",
        ),
        host: str = typer.Option(
                "0.0.0.0",
                "--host",
                "-h",
                help="监听地址",
        ),
        reload: bool = typer.Option(
                False,
                "--reload",
                "-r",
                help="是否开启热更新",
                is_flag=True,
        ),
        proxy_headers=typer.Option(
                False,
                "--proxy-headers",
                "-ph",
                help="是否开启代理",
                is_flag=True,
        ),
):
    import uvicorn

    uvicorn.run(
            "main:app",
            reload=reload,
            app_dir=settings.base_dir_str,
            host=host,
            port=port,
            forwarded_allow_ips='*',
            proxy_headers=proxy_headers,
    )


if __name__ == '__main__':
    typer.run(run)
