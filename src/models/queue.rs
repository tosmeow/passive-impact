#[derive(Clone)]
pub struct QueueEvent {
    pub queue_event: u32,
    pub queue_size: u32,
    pub time: f64,
}

#[derive(Clone)]
pub struct QueuePath {
    pub events: Vec<QueueEvent>,
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_event_creation() {
        let event = QueueEvent {
            queue_event: 1,
            queue_size: 10,
            time: 0.5,
        };
        assert_eq!(event.queue_event, 1);
        assert_eq!(event.queue_size, 10);
        assert_eq!(event.time, 0.5);
    }

    #[test]
    fn test_queue_event_clone() {
        let event = QueueEvent {
            queue_event: 2,
            queue_size: 5,
            time: 1.0,
        };
        let cloned = event.clone();
        assert_eq!(cloned.queue_event, 2);
        assert_eq!(cloned.queue_size, 5);
        assert_eq!(cloned.time, 1.0);
    }

    #[test]
    fn test_queue_path_creation() {
        let events = vec![
            QueueEvent { queue_event: 0, queue_size: 10, time: 0.0 },
            QueueEvent { queue_event: 1, queue_size: 11, time: 0.5 },
            QueueEvent { queue_event: 3, queue_size: 10, time: 1.2 },
        ];
        let path = QueuePath { events };
        assert_eq!(path.events.len(), 3);
        assert_eq!(path.events[0].queue_size, 10);
        assert_eq!(path.events[2].time, 1.2);
    }

    #[test]
    fn test_queue_path_clone() {
        let path = QueuePath {
            events: vec![
                QueueEvent { queue_event: 0, queue_size: 5, time: 0.0 },
            ],
        };
        let cloned = path.clone();
        assert_eq!(cloned.events.len(), 1);
        assert_eq!(cloned.events[0].queue_size, 5);
    }
}
