import React, { useContext, useEffect, useRef, useState } from "react";
import { DragDropProps } from "./components/interfaces";
import AppContext from "./components/interfaces";

import Topbar from "./components/Topbar"

import "./assets/scss/App.scss";
import 'bootstrap/dist/css/bootstrap.min.css';



const dragDropStyle = {
    height: '75vh', width: '75vw',
    outline: '10px dashed #b5bab6', color: '#b5bab6',
    fontSize: '5em', justifyContent: 'center', alignItems: 'center',
    borderRadius: '10px', padding: '10px', margin: 'auto',
}
const flexCenterClasses = "flex items-center justify-center";

const DragDrop = ({ loadFromFile }: DragDropProps): JSX.Element => {
    // Drag and drop for file upload
    const handleDrag = (e: any) => { e.preventDefault(); }
    const handeDrop = (e: any) => {
        e.preventDefault();
        if (e.dataTransfer.items) {
            const item = e.dataTransfer.items[0]
            if (item.kind === "file") {
                const file = item.getAsFile();
                loadFromFile(file);
            };
        };
    };
    return (
        <div style={dragDropStyle}
            onDragOver={handleDrag}
            onDrop={handeDrop}
        >
            <span >Drag microstructure file!</span>
        </div>
    )
}

const App = () => {
    const {
        image: [image, setImage],
    } = useContext(AppContext)!



    const foo = () => { }
    const fileFoo = (file: File) => { console.log('file uploaded') }

    useEffect(() => { }, [])

    return (
        <div className={`w-full h-full`}>
            <Topbar></Topbar>
            <div className={`flex`} style={{ margin: '1.5%' }} > {/*Canvas div on left, sidebar on right*/}
                {!image && <DragDrop loadDefault={foo} loadFromFile={fileFoo} />}
            </div>
        </div>
    );
};

export default App;