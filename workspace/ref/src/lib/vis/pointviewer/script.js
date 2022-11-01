// Three.js - Load .OBJ ?
// from https://r105.threejsfundamentals.org/threejs/threejs-load-obj-no-materials.html


  'use strict';

/* global THREE */

class XYZLoader extends THREE.Loader {

	load( url, onLoad, onProgress, onError ) {

		const scope = this;

		const loader = new THREE.FileLoader( this.manager );
		loader.setPath( this.path );
		loader.setRequestHeader( this.requestHeader );
		loader.setWithCredentials( this.withCredentials );
		loader.load( url, function ( text ) {

			try {

				onLoad( scope.parse( text ) );

			} catch ( e ) {

				if ( onError ) {

					onError( e );

				} else {

					console.error( e );

				}

				scope.manager.itemError( url );

			}

		}, onProgress, onError );

	}

	parse( text ) {

		const lines = text.split( '\n' );

		// const vertices = [];
		// const colors = [];

		// for ( let line of lines ) {

		// 	line = line.trim();

		// 	if ( line.charAt( 0 ) === '#' ) continue; // skip comments

		// 	const lineValues = line.split( /\s+/ );

		// 	if ( lineValues.length === 3 ) {

		// 		// XYZ

		// 		vertices.push( parseFloat( lineValues[ 0 ] ) );
		// 		vertices.push( parseFloat( lineValues[ 1 ] ) );
		// 		vertices.push( parseFloat( lineValues[ 2 ] ) );

		// 	}

		// 	if ( lineValues.length === 6 ) {

		// 		// XYZRGB

		// 		vertices.push( parseFloat( lineValues[ 0 ] ) );
		// 		vertices.push( parseFloat( lineValues[ 1 ] ) );
		// 		vertices.push( parseFloat( lineValues[ 2 ] ) );

		// 		colors.push( parseFloat( lineValues[ 3 ] ) / 255 );
		// 		colors.push( parseFloat( lineValues[ 4 ] ) / 255 );
		// 		colors.push( parseFloat( lineValues[ 5 ] ) / 255 );

		// 	}

		// }

		// const geometry = new THREE.BufferGeometry;
		// geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( vertices, 3 ) );

		// if ( colors.length > 0 ) {

		// 	geometry.setAttribute( 'color', new THREE.Float32BufferAttribute( colors, 3 ) );

		// }

		// return geometry;
        const geometry = new THREE.SphereGeometry( 15, 32, 16 );
const material = new THREE.MeshBasicMaterial( { color: 0xffff00 } );
const sphere = new THREE.Mesh( geometry, material );
return sphere;

	}

}


function readTextFile(file)
{
    var vertices = [];
    var rawFile = new XMLHttpRequest();
    rawFile.open("GET", file, false);
    rawFile.onreadystatechange = function ()
    {
        if(rawFile.readyState === 4)
        {
            if(rawFile.status === 200 || rawFile.status == 0)
            {
                var allText = rawFile.responseText;
                // alert(allText);
                vertices = textVertices(allText);
            }
        }
    }
    rawFile.send(null);
    return vertices;
}


function readTextRawFile(file)
{
    var allText = [];
    var rawFile = new XMLHttpRequest();
    rawFile.open("GET", file, false);
    rawFile.onreadystatechange = function ()
    {
        if(rawFile.readyState === 4)
        {
            if(rawFile.status === 200 || rawFile.status == 0)
            {
                allText = rawFile.responseText;
            }
        }
    }
    rawFile.send(null);
    return allText;
}


function pointcloud_scene(description, caption, url, content) {
    // create the scene
    const scene = new THREE.Scene();
    // make the background color to be white
    scene.background = new THREE.Color( 0xffffff );

    // make a list item
    const element = document.createElement( 'div' );
    element.className = 'list-item';

    const sceneElement = document.createElement( 'div' );
    sceneElement.innerText = description;
    element.appendChild( sceneElement );

    const descriptionElement = document.createElement( 'div' );
    descriptionElement.innerText = caption;
    element.appendChild( descriptionElement );

    // the element that represents the area we want to render the scene
    scene.userData.element = sceneElement;
    content.appendChild( element );

    // add camera to scene
    const camera = new THREE.PerspectiveCamera( 50, 1, 0.1, 10 );
    camera.position.z = 2;
    scene.userData.camera = camera;

    // add content to the scene
    const controls = new THREE.OrbitControls(
        scene.userData.camera, scene.userData.element 
    );
    controls.enableZoom = true;
    scene.userData.controls = controls;

    // geometry to store the spheres
    var geom = new THREE.Geometry();
    // load the vertices
    const vertices = readTextFile(url);
    // create the spheres at the position of vertices
    for ( let i = 0; i < vertices.length / 3; i++ ) {
        const geometry = new THREE.SphereGeometry( 0.005, 12, 8 )
        geometry.translate(
            vertices[3 * i], 
            vertices[3 * i + 1], 
            vertices[3 * i + 2], 
        )
        geom.mergeMesh(new THREE.Mesh( geometry ) );
    }
    const material = new THREE.MeshStandardMaterial( 
        {
            color: new THREE.Color().setHSL( 0.5, 1, 0.5 ),
            roughness: 0.5,
            metalness: 0,
            flatShading: true
        } 
    );
    scene.add( new THREE.Mesh( geom, material ) );

    scene.add( new THREE.HemisphereLight( 0xaaaaaa, 0x444444 , 1) );

    const light = new THREE.DirectionalLight( 0xffffff, 0.5 );
    light.position.set( 2, 2, 2 );

    scene.add( light );

    return scene;
}


function mesh_scene(description, caption, url, content) {
    const scene = new THREE.Scene();

    // make the background color to be white
    scene.background = new THREE.Color( 0xffffff );

    // make a list item
    const element = document.createElement( 'div' );
    element.className = 'list-item';

    const sceneElement = document.createElement( 'div' );
    sceneElement.innerText = description;
    element.appendChild( sceneElement );

    const descriptionElement = document.createElement( 'div' );
    descriptionElement.innerText = caption;
    element.appendChild( descriptionElement );

    // the element that represents the area we want to render the scene
    scene.userData.element = sceneElement;
    content.appendChild( element );

    const camera = new THREE.PerspectiveCamera( 50, 1, 0.1, 10 );
    camera.position.set(0.75, 0.75, 0.75);
    scene.userData.camera = camera;

    // add content to the scene
    const controls = new THREE.OrbitControls(
        scene.userData.camera, scene.userData.element 
    );
    controls.enableZoom = true;
    scene.userData.controls = controls;
    
    // load the obj file
    const objLoader = new THREE.OBJLoader();
    const material = new THREE.MeshStandardMaterial( 
        {
            color: new THREE.Color().setHSL( 0.04, 1, 0.75 ),
            roughness: 0.5,
            metalness: 0,
            flatShading: true
        } 
    );

    objLoader.load(
        url, 
        function( obj ){
            obj.traverse( 
                function( child ) {
                    if ( child instanceof THREE.Mesh ) {
                        child.material = material;
                    }
                }
            );
            scene.add( obj );
        },
        function( xhr ){
            console.log( (xhr.loaded / xhr.total * 100) + "% loaded")
        },
        function( err ){
            console.error( "Error loading 'ship.obj'")
        }
    );
  
    scene.add( new THREE.HemisphereLight( 0xaaaaaa, 0x444444 , 1) );

    const light = new THREE.DirectionalLight( 0xffffff, 0.5 );
    light.position.set( 2, 2, 2 );
    scene.add( light );

    return scene;
}


function textVertices(text) {
    const lines = text.split( '\n' );

    const vertices = [];
    const colors = [];

    for ( let line of lines ) {

        line = line.trim();

        if ( line.charAt( 0 ) === '#' ) continue; // skip comments

        const lineValues = line.split( /\s+/ );

        if ( lineValues.length === 3 ) {

            // XYZ

            vertices.push( parseFloat( lineValues[ 0 ] ) );
            vertices.push( parseFloat( lineValues[ 1 ] ) );
            vertices.push( parseFloat( lineValues[ 2 ] ) );

        }

        if ( lineValues.length === 6 ) {

            // XYZRGB

            vertices.push( parseFloat( lineValues[ 0 ] ) );
            vertices.push( parseFloat( lineValues[ 1 ] ) );
            vertices.push( parseFloat( lineValues[ 2 ] ) );

            colors.push( parseFloat( lineValues[ 3 ] ) / 255 );
            colors.push( parseFloat( lineValues[ 4 ] ) / 255 );
            colors.push( parseFloat( lineValues[ 5 ] ) / 255 );

        }

    }
    return vertices;
}


function main(url, cid) {
  const canvas = document.querySelector(cid);
  const renderer = new THREE.WebGLRenderer({canvas});

  const fov = 45;
  const aspect = 2;  // the canvas default
  const near = 0.1;
  const far = 100;
  const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
  camera.position.set(0.75, 0.75, 0.75);

  const controls = new THREE.OrbitControls(camera, canvas);
  controls.target.set(0, 0, 0);
  controls.update();

  const scene = new THREE.Scene();
  scene.background = new THREE.Color('white');

  {
    const skyColor = 0xB1E1FF;  // light blue
    const groundColor = 0xB97A20;  // brownish orange
    const intensity = 1;
    const light = new THREE.HemisphereLight(skyColor, groundColor, intensity);
    scene.add(light);
  }

  {
    const color = 0xFFFFFF;
    const intensity = 1;
    const light = new THREE.DirectionalLight(color, intensity);
    light.position.set(0, 10, 0);
    light.target.position.set(-5, 0, 0);
    scene.add(light);
    scene.add(light.target);
  }

  {
    const objLoader = new THREE.OBJLoader2();
    objLoader.load(url, (event) => {
      const root = event.detail.loaderRootNode;
      scene.add(root);
    });
    // https://r105.threejsfundamentals.org/threejs/resources/models/windmill/windmill.obj
    // file:///home/george/George/projects/min_modelnet/src/lib/vis/threejs/three-jsload-obj/dist/windmill.obj
  }

  function resizeRendererToDisplaySize(renderer) {
    const canvas = renderer.domElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const needResize = canvas.width !== width || canvas.height !== height;
    if (needResize) {
      renderer.setSize(width, height, false);
    }
    return needResize;
  }

  function render() {

    if (resizeRendererToDisplaySize(renderer)) {
      const canvas = renderer.domElement;
      camera.aspect = canvas.clientWidth / canvas.clientHeight;
      camera.updateProjectionMatrix();
    }

    renderer.render(scene, camera);

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
}


function abc(url, cid) {
    const canvas = document.querySelector(cid);
    const renderer = new THREE.WebGLRenderer({canvas});
  
    const fov = 45;
    const aspect = 2;  // the canvas default
    const near = 0.1;
    const far = 100;
    const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    camera.position.set(0.75, 0.75, 0.75);
  
    const controls = new THREE.OrbitControls(camera, canvas);
    controls.target.set(0, 0, 0);
    controls.update();
  
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('white');
  
    {
      const skyColor = 0xB1E1FF;  // light blue
      const groundColor = 0xB97A20;  // brownish orange
      const intensity = 1;
      const light = new THREE.HemisphereLight(skyColor, groundColor, intensity);
      scene.add(light);
    }
  
    {
      const color = 0xFFFFFF;
      const intensity = 1;
      const light = new THREE.DirectionalLight(color, intensity);
      light.position.set(0, 10, 0);
      light.target.position.set(-5, 0, 0);
      scene.add(light);
      scene.add(light.target);
    }
  
    {

    //   const objLoader = new XYZLoader();
    //   objLoader.load(url, (event) => {
    //     const root = event.detail.loaderRootNode;
    //     scene.add(root);
    //   });
    const vertices = readTextFile('air_0001.xyz');
//     let fs = require('fs');
// let all = fs.readFileSync('helix_201.xyz', "utf8");
// all = all.trim();  // final crlf in file
//     const lines = all.split( '\n' );

// 		const vertices = [];
// 		const colors = [];

// 		for ( let line of lines ) {

// 			line = line.trim();

// 			if ( line.charAt( 0 ) === '#' ) continue; // skip comments

// 			const lineValues = line.split( /\s+/ );

// 			if ( lineValues.length === 3 ) {

// 				// XYZ

// 				vertices.push( parseFloat( lineValues[ 0 ] ) );
// 				vertices.push( parseFloat( lineValues[ 1 ] ) );
// 				vertices.push( parseFloat( lineValues[ 2 ] ) );

// 			}

// 			if ( lineValues.length === 6 ) {

// 				// XYZRGB

// 				vertices.push( parseFloat( lineValues[ 0 ] ) );
// 				vertices.push( parseFloat( lineValues[ 1 ] ) );
// 				vertices.push( parseFloat( lineValues[ 2 ] ) );

// 				colors.push( parseFloat( lineValues[ 3 ] ) / 255 );
// 				colors.push( parseFloat( lineValues[ 4 ] ) / 255 );
// 				colors.push( parseFloat( lineValues[ 5 ] ) / 255 );

// 			}

// 		}
// console.log( vertices );
    for ( let i = 0; i < vertices.length / 3; i++ ) {
        const geometries = [
            // new THREE.BoxGeometry( 0.01, 1, 1 ),
            new THREE.SphereGeometry( 0.005, 12, 8 ),
            // new THREE.DodecahedronGeometry( 0.005 ),
            // new THREE.CylinderGeometry( 0.005, 0.5, 1, 12 )
        ];
        const geometry = geometries[ geometries.length * Math.random() | 0 ];
    
                        const material = new THREE.MeshStandardMaterial( {
    
                            color: new THREE.Color().setHSL( Math.random(), 1, 0.75 ),
                            roughness: 0.5,
                            metalness: 0,
                            flatShading: true
    
                        } );
        console.log( geometry.center() );
        geometry.translate(
            vertices[3 * i], 
            vertices[3 * i + 1], 
            vertices[3 * i + 2], 
        )
    
                        scene.add( new THREE.Mesh( geometry, material ) );

    }

    
    // const geometry = new THREE.SphereGeometry( 5, 32, 16 );
    // const material = new THREE.MeshBasicMaterial( { color: 0xffff00 } );
    // const sphere = new THREE.Mesh( geometry, material );
    // sphere.position.set ( 0, 0, 0 );
    // scene.add( sphere );
      // https://r105.threejsfundamentals.org/threejs/resources/models/windmill/windmill.obj
      // file:///home/george/George/projects/min_modelnet/src/lib/vis/threejs/three-jsload-obj/dist/windmill.obj
    }
  
    function resizeRendererToDisplaySize(renderer) {
      const canvas = renderer.domElement;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const needResize = canvas.width !== width || canvas.height !== height;
      if (needResize) {
        renderer.setSize(width, height, false);
      }
      return needResize;
    }
  
    function render() {
  
      if (resizeRendererToDisplaySize(renderer)) {
        const canvas = renderer.domElement;
        camera.aspect = canvas.clientWidth / canvas.clientHeight;
        camera.updateProjectionMatrix();
      }
  
      renderer.render(scene, camera);
  
      requestAnimationFrame(render);
    }
  
    requestAnimationFrame(render);
}


function text_scene(description, caption, url, content){
    const scene = new THREE.Scene();

    // make the background color to be white
    scene.background = new THREE.Color( 0xffffff );

    // make a list item
    const element = document.createElement( 'div' );
    element.className = 'list-item';

    const sceneElement = document.createElement( 'div' );
    sceneElement.innerText = description + ': ' + readTextRawFile(url);
    element.appendChild( sceneElement );

    const descriptionElement = document.createElement( 'div' );
    descriptionElement.innerText = caption;
    element.appendChild( descriptionElement );

    // the element that represents the area we want to render the scene
    scene.userData.element = sceneElement;
    content.appendChild( element );

    // add camera to scene
    const camera = new THREE.PerspectiveCamera( 50, 1, 0.1, 10 );
    camera.position.z = 2;
    scene.userData.camera = camera;

    return scene;
}


function create_scenes(description_arr, caption_arr, url_arr, type_arr) {
    const scenes = [];
    // get the canvas
    let canvas;
    canvas = document.getElementById( "c" );
    // get the content
    const content = document.getElementById( 'content' );
    for (let i = 0; i < description_arr.length; i++) {
        if (type_arr[i] == 'pointcloud'){
            const scene = pointcloud_scene(
                description_arr[i], 
                caption_arr[i], 
                url_arr[i], content);
            scenes.push( scene );
        }
        if (type_arr[i] == 'mesh') {
            const scene = mesh_scene(
                description_arr[i], 
                caption_arr[i], 
                url_arr[i], content);
            scenes.push( scene );
        }
        if (type_arr[i] == 'text') {
            const scene = text_scene(
                description_arr[i], 
                caption_arr[i], 
                url_arr[i], content);
            scenes.push( scene );
        }
    }
    return scenes;
}


function scenes_render(description_arr, caption_arr, url_arr, type_arr){
    let canvas, renderer;

    canvas = document.getElementById( "c" );

    var scenes = create_scenes(description_arr, caption_arr, url_arr, type_arr);

    renderer = new THREE.WebGLRenderer( { canvas: canvas, antialias: true } );
    renderer.setClearColor( 0xffffff, 1 );
    renderer.setPixelRatio( window.devicePixelRatio );

    animate();

    function updateSize() {

        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
    
        if ( canvas.width !== width || canvas.height !== height ) {
    
            renderer.setSize( width, height, false );
    
        }
    
    }

    function animate() {
    
        render();
        requestAnimationFrame( animate );
    
    }

    function render() {

        updateSize(renderer);
    
        canvas.style.transform = `translateY(${window.scrollY}px)`;
    
        renderer.setClearColor( 0xffffff );
        renderer.setScissorTest( false );
        renderer.clear();
    
        renderer.setClearColor( 0xe0e0e0 );
        renderer.setScissorTest( true );
    
        scenes.forEach( function ( scene ) {
            // get the element that is a place holder for where we want to
            // draw the scene
            const element = scene.userData.element;
    
            // get its position relative to the page's viewport
            const rect = element.getBoundingClientRect();
    
            // check if it's offscreen. If so skip it
            if ( rect.bottom < 0 || rect.top > renderer.domElement.clientHeight ||
                 rect.right < 0 || rect.left > renderer.domElement.clientWidth ) {
    
                return; // it's off screen
    
            }
    
            // set the viewport
            const width = rect.right - rect.left;
            const height = rect.bottom - rect.top;
            const left = rect.left;
            const bottom = renderer.domElement.clientHeight - rect.bottom;
    
            renderer.setViewport( left, bottom, width, height );
            renderer.setScissor( left, bottom, width, height );
    
            const camera = scene.userData.camera;
    
            renderer.render( scene, camera );
        } );
    }
}
