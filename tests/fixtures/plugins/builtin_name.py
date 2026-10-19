from stt_daemon.plugin import Backend, Options, Result


class Shadow(Backend):
    name = "parakeet"

    async def transcribe(self, pcm: bytes, options: Options) -> Result:
        return Result("shadow")
