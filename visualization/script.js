import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { pipeline, env } from '@xenova/transformers';

// --- Configuration ---
env.allowLocalModels = false;
env.useBrowserCache = true;

const CONFIG = {
    pointRadius: 0.3,
    pointColor: 0x0066cc, // Darker blue for visibility on white
    selectedColor: 0xff0055, // Vivid red/pink
    lineColor: 0x333333, // Dark grey lines
    lineOpacity: 0.4,
    embeddingScale: 2, // t-SNE output needs less scaling than our pseudo-random one
    arrowColor: 0xff8800, // Orange arrows
    arrowLength: 1.5,
};

// --- Helper Functions ---

// Data Loading Helper
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

// UI Helper for Loading
function updateLoadingStatus(message) {
    const tooltip = document.getElementById('tooltip');
    if (message) {
        tooltip.innerHTML = message;
        tooltip.classList.remove('hidden');
        tooltip.style.left = '50%';
        tooltip.style.top = '50%';
        tooltip.style.transform = 'translate(-50%, -50%)';
        tooltip.style.zIndex = '1000';
    } else {
        tooltip.classList.add('hidden');
        tooltip.style.transform = 'translate(-50%, -100%)'; // Reset to default tooltip behavior
        tooltip.style.left = '0';
        tooltip.style.top = '0';
        tooltip.style.zIndex = '10';
    }
}

// Embedding Generation
let featureExtractor = null;

async function generateEmbeddings(texts) {
    try {
        if (!featureExtractor) {
            updateLoadingStatus("Loading embedding model (Xenova/all-MiniLM-L6-v2)...<br>This happens in your browser and may take a moment.");
            featureExtractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        }

        updateLoadingStatus("Generating embeddings...");
        const embeddings = [];
        
        for (let i = 0; i < texts.length; i++) {
            // Update progress occasionally
            updateLoadingStatus(`Generating embeddings: ${i + 1}/${texts.length}`);
            
            // Generate embedding
            const output = await featureExtractor(texts[i], { pooling: 'mean', normalize: true });
            embeddings.push(Array.from(output.data));
        }
        
        return embeddings;
    } catch (e) {
        console.error("Embedding generation failed:", e);
        alert("Failed to load model or generate embeddings. See console.");
        return [];
    }
}

// t-SNE Projection
async function runTSNE(embeddings) {
    updateLoadingStatus("Running t-SNE projection...");
    
    return new Promise((resolve) => {
        setTimeout(() => {
            try {
                // Initialize t-SNE
                const opt = {
                    epsilon: 10, 
                    perplexity: Math.max(5, Math.min(30, embeddings.length - 1)), 
                    dim: 3 
                };
                
                // tsnejs usually exposes 'tsne' object with 'tSNE' constructor
                const T = new window.tsne.tSNE(opt); 
                T.initDataRaw(embeddings);
                
                // Run optimization
                const iterations = 500;
                for (let k = 0; k < iterations; k++) {
                    T.step();
                }
                
                resolve(T.getSolution()); // Returns array of [x, y, z]
            } catch (e) {
                console.error("t-SNE failed:", e);
                // Fallback to random if t-SNE fails
                resolve(embeddings.map(() => [Math.random(), Math.random(), Math.random()]));
            }
        }, 100); // Small delay to let UI render the message
    });
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
const interactableObjects = []; 
let selectedObject = null;
let hoveredObject = null;

// --- Build Visualization ---
function buildScene(trajectoryData) {
    // trajectoryData is array of { id, problem, method, conclusion, position: THREE.Vector3 }
    
    // Clear previous scene elements if any (not strictly necessary for single run but good practice)
    // ...

    const nodeGeometry = new THREE.SphereGeometry(CONFIG.pointRadius, 32, 32);
    const nodeMaterial = new THREE.MeshStandardMaterial({ 
        color: CONFIG.pointColor,
        roughness: 0.3,
        metalness: 0.5,
        emissive: 0x000000 
    });
    
    const lineMaterial = new THREE.LineBasicMaterial({ 
        color: CONFIG.lineColor, 
        transparent: true, 
        opacity: CONFIG.lineOpacity 
    });

    // We assume single trajectory for now based on previous implementation
    const points = [];
    const group = new THREE.Group();
    scene.add(group);

    trajectoryData.forEach((step) => {
        const { x, y, z } = step.position;

        const mesh = new THREE.Mesh(nodeGeometry, nodeMaterial.clone());
        mesh.position.set(x, y, z);
        mesh.userData = { ...step, originalColor: CONFIG.pointColor }; 
        
        group.add(mesh);
        interactableObjects.push(mesh);
        points.push(new THREE.Vector3(x, y, z));
    });

    // Draw lines and arrows
    if (points.length > 1) {
        const curve = new THREE.CatmullRomCurve3(points);
        const pointsCount = points.length * 10;
        const curvePoints = curve.getPoints(pointsCount);
        const curveGeometry = new THREE.BufferGeometry().setFromPoints(curvePoints);
        const curveLine = new THREE.Line(curveGeometry, lineMaterial);
        group.add(curveLine);

        for (let i = 0; i < points.length - 1; i++) {
            const startPoint = points[i];
            const endPoint = points[i + 1];
            
            const direction = new THREE.Vector3().subVectors(endPoint, startPoint).normalize();
            
            const arrowOrigin = startPoint.clone().add(direction.clone().multiplyScalar(CONFIG.pointRadius * 1.2));
            const distance = startPoint.distanceTo(endPoint);
            const maxLen = distance - (CONFIG.pointRadius * 2.5);
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
    
    // Center camera on the centroid
    if (points.length > 0) {
        const center = new THREE.Vector3();
        points.forEach(p => center.add(p));
        center.divideScalar(points.length);
        controls.target.copy(center);
        camera.lookAt(center);
    }
}

// --- Main execution ---
async function init() {
    readTextFile("/api/data", async function(text) {
        try {
            const rawData = JSON.parse(text);
            
            // Prepare texts for embedding
            // Concatenate relevant fields to get a meaningful representation
            const texts = rawData.map(step => `${step.p}\n${step.s}\n${step.c}`);
            
            // 1. Generate Embeddings
            const embeddings = await generateEmbeddings(texts);
            
            // 2. Project with t-SNE
            const projectedPoints = await runTSNE(embeddings);
            
            updateLoadingStatus(""); // Clear loading message

            // 3. Process data for scene
            const trajectoryData = rawData.map((step, index) => {
                const coords = projectedPoints[index];
                return {
                    id: `step-${index}`,
                    problem: step.p,
                    method: step.s,
                    conclusion: step.c,
                    position: new THREE.Vector3(
                        coords[0] * CONFIG.embeddingScale, 
                        coords[1] * CONFIG.embeddingScale, 
                        coords[2] * CONFIG.embeddingScale
                    )
                };
            });

            buildScene(trajectoryData);

        } catch (e) {
            console.error("Error processing data:", e);
            alert("Error processing data.");
        }
    });
}

init();

// --- Interaction (Same as before) ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const infoPanel = document.getElementById('info-panel');
const tooltip = document.getElementById('tooltip');
const closeBtn = document.getElementById('close-panel');

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
    }
}

window.addEventListener('click', onMouseClick);
window.addEventListener('mousemove', onMouseMove);

function onMouseMove(event) {
    // Only update hover if not showing global loading message
    if (!tooltip.innerHTML.includes("Loading") && !tooltip.innerHTML.includes("Generating")) {
        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(interactableObjects);

        if (intersects.length > 0) {
            const object = intersects[0].object;
            if (hoveredObject !== object) {
                hoveredObject = object;
                tooltip.classList.remove('hidden');
                // Use substring for preview
                const problemShort = object.userData.problem.length > 50 ? object.userData.problem.substring(0, 50) + "..." : object.userData.problem;
                tooltip.innerHTML = `<strong>${problemShort}</strong>`;
            }
            tooltip.style.left = event.clientX + 'px';
            tooltip.style.top = event.clientY + 'px';
            tooltip.style.transform = 'translate(-50%, -100%)';
        } else {
            if (hoveredObject) {
                hoveredObject = null;
                tooltip.classList.add('hidden');
            }
        }
    }
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

animate();