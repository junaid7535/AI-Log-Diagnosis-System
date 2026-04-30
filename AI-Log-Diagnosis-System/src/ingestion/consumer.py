import asyncio
from typing import Callable, Optional
from kafka import KafkaConsumer
from kafka.structs import ConsumerRecord
import json
from loguru import logger
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LogEntry:
    raw: str
    timestamp: datetime
    source: str
    metadata: dict

class LogConsumer:
    def __init__(self, topic: str, bootstrap_servers: list, processor: Callable):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=False,
            max_poll_records=1000,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        self.processor = processor
        self.running = False
        
    async def start(self):
        self.running = True
        logger.info(f"Starting consumer on topic")
        while self.running:
            msg_batch = self.consumer.poll(timeout_ms=1000, max_records=500)
            for tp, messages in msg_batch.items():
                for msg in messages:
                    await self._handle_message(msg)
                self.consumer.commit()
            await asyncio.sleep(0.01)
    
    async def _handle_message(self, msg: ConsumerRecord):
        try:
            log_entry = LogEntry(
                raw=msg.value.get('message', ''),
                timestamp=datetime.fromtimestamp(msg.timestamp/1000),
                source=msg.value.get('source', 'unknown'),
                metadata=msg.value.get('metadata', {})
            )
            await self.processor(log_entry)
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
    
    def stop(self):
        self.running = False
        self.consumer.close()