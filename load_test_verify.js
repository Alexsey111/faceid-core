import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
    stages: [
        { duration: '1m', target: 5 },
        { duration: '1m', target: 10 },
        { duration: '1m', target: 10 },
        { duration: '30s', target: 0 },
    ],
};

const BASE_URL = 'http://localhost:8000';

// Real base64 generated from tests/data/person1_small.jpg
const raw = open('./tests/data/person1_small.b64.txt');
const IMAGE_BASE64 = raw
    .replace(/\r/g, '')
    .replace(/\n/g, '')
    .trim();

export default function () {
    const payload = JSON.stringify({
        user_id: "1",
        image: IMAGE_BASE64,
        require_liveness: false
    });

    const headers = {
        'Content-Type': 'application/json',
    };

    const res = http.post(`${BASE_URL}/verify_base64`, payload, {
        headers,
        timeout: '10s',
    });

    if (res.status !== 200) {
        console.log(`ERROR STATUS: ${res.status}`);
    }

    check(res, {
        'status is 200': (r) => r.status === 200,
        'body not empty': (r) => r.body && r.body.length > 0,
        'has status field': (r) => JSON.parse(r.body).status !== undefined,
    });

    sleep(0.2);
}
