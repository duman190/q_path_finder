OPENQASM 2.0;
include "qelib1.inc";
// DQI l=1, instance B1: m=3 metrics, n=2 stages, v=[1, 0, 0]
// qubits 0..2 metric, 3..4 path (stage j = q[3+j])
// c[j] = port of stage j; Qiskit count strings print c[1] first (stage j = char n-1-j)
qreg q[5];
creg c[2];
ry(1.0471975511965979) q[0];
ry(0.95531661812450919) q[1];
cx q[0],q[1];
ry(-0.95531661812450919) q[1];
cx q[0],q[1];
cx q[1],q[0];
ry(0.78539816339744839) q[2];
cx q[1],q[2];
ry(-0.78539816339744839) q[2];
cx q[1],q[2];
cx q[2],q[1];
z q[0];
cx q[0],q[3];
cx q[1],q[4];
cx q[2],q[3];
cx q[2],q[4];
x q[4];
ccx q[3],q[4],q[0];
x q[4];
x q[3];
ccx q[3],q[4],q[1];
x q[3];
ccx q[3],q[4],q[2];
h q[3];
h q[4];
measure q[3] -> c[0];
measure q[4] -> c[1];
