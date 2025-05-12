from confluent_kafka import Consumer, KafkaError
import json

# Kafka configuration
bootstrap_servers = 'localhost:9092'
topic = 'audio_recommendations'
group_id = 'recommendation_consumer'

# Kafka consumer configuration
config = {
    'bootstrap.servers': bootstrap_servers,
    'group.id': group_id,
    'auto.offset.reset': 'earliest'
}

# Create Kafka consumer
consumer = Consumer(config)
consumer.subscribe([topic])

# Function to consume recommendation messages
def consume_recommendations(consumer):
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(msg.error())
                    break
            # Process recommendation messages
            recommendations = json.loads(msg.value())
            for recommendation in recommendations:
                  print(f"Randomly Chosen File: {recommendation['randomly_chosen_file']} - Recommended Track: {recommendation['recommended_track']} - Similarity Score: {recommendation['similarity_score']}")
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

# Consume recommendation messages
consume_recommendations(consumer)

