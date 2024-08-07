import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext from "./interfaces";

import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import { TopbarProps } from "./interfaces";

const Topbar = ({ loadFromFile, reset, changePhase }: TopbarProps) => {

    const {
        showInfo: [showInfo, setShowInfo],
    } = useContext(AppContext)!;
    const fileInputRef = useRef<HTMLInputElement>(null);

    const addData = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click();
        };
    }

    const handleFileUpload = (e: any) => {
        // Open file dialog and load file.
        const file: File | null = e.target.files?.[0] || null;
        if (file != null) {
            loadFromFile(file);
        };
    }

    return (
        <Navbar bg={"dark"} variant="dark" expand="lg" style={{ boxShadow: "1px 1px  1px grey" }}>
            <Container>
                {/*path for these assets need to be relative to index.html in assets/*/}
                <Navbar.Brand><img src="favicon.png" width="40" height="40" className="d-inline-block align-top" /></Navbar.Brand>
                <Navbar.Brand style={{ marginTop: "3px", fontSize: "1.75em" }}>ImageRep</Navbar.Brand>

                <Nav>
                    <Nav.Link onClick={e => setShowInfo(true)}>Model Info</Nav.Link>
                    <Nav.Link onClick={addData}>Add Data</Nav.Link>
                    <Nav.Link onClick={changePhase} style={{ color: "#f2cd29" }}>Change Phase</Nav.Link>
                    <input
                        type='file'
                        id='file_load'
                        ref={fileInputRef}
                        style={{ display: 'none' }}
                        onChange={e => handleFileUpload(e)} />
                    <Nav.Link style={{ color: "red" }} onClick={(e) => reset()}>Reset</Nav.Link>
                </Nav>

            </Container>
        </Navbar >
    );
}

export default Topbar;