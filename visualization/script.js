import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- Configuration ---
const CONFIG = {
    pointRadius: 0.3,
    pointColor: 0x0066cc, // Darker blue for visibility on white
    selectedColor: 0xff0055, // Vivid red/pink
    lineColor: 0x333333, // Dark grey lines
    lineOpacity: 0.4,
    embeddingScale: 15, // Scale for the embedding coordinates
    arrowColor: 0xff8800, // Orange arrows
    arrowLength: 1.5,
};

// --- Pseudo-Embedding Helper ---
// Simulates mapping a text string to a vector in 3D space.
// In a real application, this would be replaced by an API call or precomputed embeddings (e.g., BERT/GPT -> PCA/t-SNE -> 3D).
function getPseudoEmbedding(text) {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
        hash = ((hash << 5) - hash) + text.charCodeAt(i);
        hash |= 0; // Convert to 32bit integer
    }
    // Use the hash to seed a pseudo-random number generator for 3 dimensions
    // We use a simple sin/cos based generic mapping to distribute points in a sphere-like volume
    const seed = Math.abs(hash);
    const x = (Math.sin(seed * 0.123) * CONFIG.embeddingScale);
    const y = (Math.cos(seed * 0.456) * CONFIG.embeddingScale);
    const z = (Math.sin(seed * 0.789) * CONFIG.embeddingScale);
    
    return new THREE.Vector3(x, y, z);
}

// --- Data Loading Helper ---
function readTextFile(file, callback) {
    var rawFile = new XMLHttpRequest();
    rawFile.overrideMimeType("application/json");
    rawFile.open("GET", file, true);
    rawFile.onreadystatechange = function() {
        if (rawFile.readyState === 4 && rawFile.status == "200") {
            callback(rawFile.responseText);
        }
    }
    rawFile.send(null);
}

// --- Scene Setup ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf0f0f0); // White/Light Grey background
scene.fog = new THREE.FogExp2(0xf0f0f0, 0.02);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 5, 20);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.getElementById('container').appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// --- Lighting ---
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const pointLight = new THREE.PointLight(0xffffff, 1);
pointLight.position.set(10, 10, 10);
scene.add(pointLight);

// --- Visualization State ---
const interactableObjects = []; // Stores meshes for raycasting
let selectedObject = null;
let hoveredObject = null;

// --- Build Visualization ---
function buildScene(data) {
    const nodeGeometry = new THREE.SphereGeometry(CONFIG.pointRadius, 32, 32);
    const nodeMaterial = new THREE.MeshStandardMaterial({ 
        color: CONFIG.pointColor,
        roughness: 0.3,
        metalness: 0.5,
        emissive: 0x000000 // Reduced emissive for light background
    });
    
    const lineMaterial = new THREE.LineBasicMaterial({ 
        color: CONFIG.lineColor, 
        transparent: true, 
        opacity: CONFIG.lineOpacity 
    });

    data.forEach((trajectory, tIndex) => {
        const points = [];
        
        // Group for this trajectory
        const group = new THREE.Group();
        scene.add(group);

        // Draw nodes
        trajectory.forEach((step, sIndex) => {
            // Use the embedding position stored in the data
            const { x, y, z } = step.position;

            const mesh = new THREE.Mesh(nodeGeometry, nodeMaterial.clone());
            mesh.position.set(x, y, z);
            mesh.userData = { ...step, originalColor: CONFIG.pointColor }; // Store data in mesh
            
            group.add(mesh);
            interactableObjects.push(mesh);
            points.push(new THREE.Vector3(x, y, z));
        });

        // Draw lines and arrows connecting nodes
        if (points.length > 1) {
            // Curve
            const curve = new THREE.CatmullRomCurve3(points);
            const pointsCount = points.length * 10;
            const curvePoints = curve.getPoints(pointsCount);
            const curveGeometry = new THREE.BufferGeometry().setFromPoints(curvePoints);
            const curveLine = new THREE.Line(curveGeometry, lineMaterial);
            group.add(curveLine);

            // Arrows at each node (except the last one) pointing to the next node
            for (let i = 0; i < points.length - 1; i++) {
                const startPoint = points[i];
                const endPoint = points[i + 1];
                
                const direction = new THREE.Vector3().subVectors(endPoint, startPoint).normalize();
                
                // Offset arrow slightly from the node so it's visible
                // Start outside the sphere (radius is CONFIG.pointRadius)
                const arrowOrigin = startPoint.clone().add(direction.clone().multiplyScalar(CONFIG.pointRadius * 1.2));
                
                // Adjust length based on distance, but keep it reasonable
                const distance = startPoint.distanceTo(endPoint);
                // Make arrow bigger and ensure it fits in the segment
                const maxLen = distance - (CONFIG.pointRadius * 2.5); // Ensure it doesn't hit the next node
                const length = Math.min(CONFIG.arrowLength * 1.5, Math.max(0.5, maxLen)); 
                
                if (length > 0.1) {
                    const headLength = length * 0.4;
                    const headWidth = headLength * 0.6;

                    const arrowHelper = new THREE.ArrowHelper(
                        direction, 
                        arrowOrigin, 
                        length, 
                        CONFIG.arrowColor, 
                        headLength, 
                        headWidth
                    );
                    group.add(arrowHelper);
                }
            }
        }
    });
}

// --- Load Data and Build Scene ---
readTextFile("../agents/traj.json", function(text) {
    const rawTrajectoryData = JSON.parse(text);
    
    // Treat the linear list of steps as a single trajectory for visualization
    const trajectory = rawTrajectoryData.map((step, index) => {
        const problem = step.p;
        const method = step.s;
        const conclusion = step.c;
        
        // Generate embedding coordinate based on the content
        const content = `${problem} ${method} ${conclusion}`;
        const position = getPseudoEmbedding(content);

        return {
            id: `step-${index}`,
            problem: problem,
            method: method,
            conclusion: conclusion,
            position: position
        };
    });

    const trajectories = [trajectory];
    buildScene(trajectories);
});

// --- Interaction ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const infoPanel = document.getElementById('info-panel');
const tooltip = document.getElementById('tooltip');
const closeBtn = document.getElementById('close-panel');

// UI Elements
const uiProblem = document.getElementById('point-problem');
const uiMethod = document.getElementById('point-method');
const uiConclusion = document.getElementById('point-conclusion');

function updateInfoPanel(data) {
    uiProblem.textContent = data.problem;
    uiMethod.textContent = data.method;
    uiConclusion.textContent = data.conclusion;
    infoPanel.classList.remove('hidden');
}

function resetSelection() {
    if (selectedObject) {
        selectedObject.material.color.setHex(selectedObject.userData.originalColor);
        selectedObject.scale.set(1, 1, 1);
        selectedObject = null;
    }
}

closeBtn.addEventListener('click', () => {
    infoPanel.classList.add('hidden');
    resetSelection();
});

function onMouseClick(event) {
    // Calculate mouse position in normalized device coordinates
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    const intersects = raycaster.intersectObjects(interactableObjects);

    if (intersects.length > 0) {
        const object = intersects[0].object;
        
        if (selectedObject !== object) {
            resetSelection();
            
            selectedObject = object;
            selectedObject.material.color.setHex(CONFIG.selectedColor);
            selectedObject.scale.set(1.5, 1.5, 1.5);
            
            updateInfoPanel(object.userData);
        }
    } else {
        // Optional: Deselect if clicking empty space
        // resetSelection();
        // infoPanel.classList.add('hidden');
    }
}

window.addEventListener('click', onMouseClick);
window.addEventListener('mousemove', onMouseMove);

function onMouseMove(event) {
    // Calculate mouse position in normalized device coordinates
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    const intersects = raycaster.intersectObjects(interactableObjects);

    if (intersects.length > 0) {
        const object = intersects[0].object;
        
        if (hoveredObject !== object) {
            hoveredObject = object;
            
            // Show tooltip
            tooltip.classList.remove('hidden');
            // Truncate tooltip content if too long
            const problemShort = object.userData.problem.length > 50 ? object.userData.problem.substring(0, 50) + "..." : object.userData.problem;
            tooltip.innerHTML = `<strong>${problemShort}</strong>`;
            
            // Optional: slight highlight if not selected
            if (object !== selectedObject) {
                 // We could change color here, but let's keep it simple for now or maybe scale slightly
                 // object.scale.set(1.2, 1.2, 1.2); 
            }
        }

        // Update tooltip position
        tooltip.style.left = event.clientX + 'px';
        tooltip.style.top = event.clientY + 'px';

    } else {
        if (hoveredObject) {
            // Reset hover state
            // if (hoveredObject !== selectedObject) hoveredObject.scale.set(1, 1, 1);
            hoveredObject = null;
            tooltip.classList.add('hidden');
        }
    }
}

// Handle Window Resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- Animation Loop ---
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

animate();