from stt_daemon.plugin import Backend, Options, Result


class First(Backend):
    name = "first"

    async def transcribe(self, pcm: bytes, options: Options) -> Result:
        return Result("one")


class Second(Backend):
    name = "second"

    async def transcribe(self, pcm: bytes, options: Options) -> Result:
        return Result("two")
