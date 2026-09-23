import "../styles/Sidebar.css"
import lookingLaptop from "../assets/Looking at laptop.png";
import lookingUser from "../assets/Looking at us.png";

import { FaLinkedin, FaGithub } from "react-icons/fa";
import { MdEmail, MdWeb } from "react-icons/md";
import { HiOutlineGlobeAlt } from "react-icons/hi2";

import { useEffect, useState } from "react";

function Sidebar() {
    const [currentImage, setCurrentImage] = useState(lookingLaptop);
    useEffect(() => {

        let timeout;

        function switchImages() {

            setCurrentImage(lookingUser);

            timeout = setTimeout(() => {

                setCurrentImage(lookingLaptop);

            }, 2000);

        }

        const interval = setInterval(switchImages, 10000);

        return () => {

            clearInterval(interval);
            clearTimeout(timeout);

        };

    }, []);
    return (
        <aside className="sidebar">

            <div className="sidebar__top">

                <h1 className="sidebar__title">
                    AskFatima
                </h1>

                <p className="sidebar__subtitle">
                    AI Portfolio Assistant
                </p>

                <p className="sidebar__description">
                    Chat with an AI version of Fatima to learn about her
                    projects, experience, education, technical skills, and
                    career goals.
                </p>

            </div>
            <div className="sidebar__image">
                <img
                    src={currentImage}
                    alt="AskFatima"
                />

            </div>


            <div className="sidebar__footer">
                <a
                    href="https://www.linkedin.com/in/fatima-iboubkarne-849675281/"
                    target="_blank"
                    rel="noreferrer"
                    aria-label="LinkedIn"
                >
                    <FaLinkedin />
                </a>

                <a
                    href="https://github.com/fatimIB"
                    target="_blank"
                    rel="noreferrer"
                    aria-label="GitHub"
                >
                    <FaGithub />
                </a>

                <a
                    href="https://fatima-iboubkarne.vercel.app/"
                    target="_blank"
                    rel="noreferrer"
                    aria-label="Portfolio"
                >
                    <MdWeb />
                </a>

                <a
                    href="mailto:fatima.iboubkarne.pro@email.com"
                    aria-label="Email"
                >
                    <MdEmail />
                </a>
            </div>

        </aside>
    );
}

export default Sidebar;