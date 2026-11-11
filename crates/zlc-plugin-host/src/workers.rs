//! Threads that stay, for work that may wait but arrives often.
//!
//! A node's device calls come tens of times a second and any one of them may wait on
//! another plugin, so none may run on the channel's thread or behind another. A new
//! thread for each was measured to start as late as 90 ms after it was asked for on
//! Windows, against 0.3 ms usually: a job here goes to a thread already waiting, and a
//! new one starts only when every thread is busy.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

type Job = Box<dyn FnOnce() + Send>;

/// How long a thread waits for work before it ends; a job after that starts another.
const IDLE: Duration = Duration::from_secs(60);

#[derive(Default)]
struct Queue { jobs: VecDeque<Job>, idle: usize }

#[derive(Default)]
struct Shared { queue: Mutex<Queue>, ready: Condvar }

pub struct Workers { name: &'static str, shared: Arc<Shared> }

impl Workers {
    pub fn new(name: &'static str) -> Self { Self { name, shared: Arc::default() } }

    /// Runs `job` now on a thread of its own: never behind a job that is waiting.
    pub fn run(&self, job: impl FnOnce() + Send + 'static) {
        let start = {
            let mut queue = self.shared.queue.lock().unwrap();
            queue.jobs.push_back(Box::new(job));
            queue.jobs.len() > queue.idle
        };
        if start {
            let shared = self.shared.clone();
            let _ = std::thread::Builder::new().name(self.name.into()).spawn(move || work(&shared));
        } else {
            self.shared.ready.notify_one();
        }
    }
}

fn work(shared: &Shared) {
    let mut queue = shared.queue.lock().unwrap();
    loop {
        if let Some(job) = queue.jobs.pop_front() {
            drop(queue);
            job();
            queue = shared.queue.lock().unwrap();
            continue;
        }
        queue.idle += 1;
        let (next, waited) = shared.ready.wait_timeout(queue, IDLE).unwrap();
        queue = next;
        queue.idle -= 1;
        if waited.timed_out() && queue.jobs.is_empty() { return; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn a_waiting_job_holds_no_other_and_a_free_thread_is_used_again() {
        let workers = Workers::new("zlc-test-worker");
        let (release, released) = mpsc::channel::<()>();
        let (started, start) = mpsc::channel();
        let told = started.clone();
        workers.run(move || { told.send(std::thread::current().id()).unwrap(); released.recv().unwrap(); });
        let blocked = start.recv_timeout(Duration::from_secs(5)).unwrap();
        // The first is waiting: the second runs beside it.
        let told = started.clone();
        workers.run(move || told.send(std::thread::current().id()).unwrap());
        let beside = start.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_ne!(blocked, beside);
        // Once that thread is waiting for work, the next job is its own.
        let until = std::time::Instant::now() + Duration::from_secs(5);
        while workers.shared.queue.lock().unwrap().idle == 0 && std::time::Instant::now() < until { std::thread::yield_now(); }
        let told = started.clone();
        workers.run(move || told.send(std::thread::current().id()).unwrap());
        assert_eq!(start.recv_timeout(Duration::from_secs(5)).unwrap(), beside);
        release.send(()).unwrap();
    }
}
